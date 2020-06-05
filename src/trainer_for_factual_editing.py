
import numpy as np
import torch
import logging
import os
import json

from transformers import (
    PreTrainedModel,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollator
)
from tqdm.auto import tqdm, trange
from typing import Optional, Any, Dict, List, NewType, Tuple, Callable, NamedTuple
from torch.utils.data import Dataset, DataLoader
from contextlib import contextmanager

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex

'''
This Trainer is modified based on 'transformers' package Trainer. 
'''

logger = logging.getLogger(__name__)

PREFIX_CHECKPOINT_DIR = "checkpoint"
START_OF_ENTITY_index = 30522
END_OF_ENTITY_index = 30523

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for the first one (locally) to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]

class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float


class TrainerForFactualEditing(Trainer):

    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
    ):
        super().__init__(
            model = model, 
            args = args, 
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            compute_metrics = compute_metrics, 
            prediction_loss_only = prediction_loss_only
        )

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        model.to(self.args.device)
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0],
        )
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if self.args.local_rank in [-1, 0]:
                        if (self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0) or (
                            global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            if self.args.evaluate_during_training:
                                results = self.evaluate()
                                for key, value in results.items():
                                    eval_key = "eval_{}".format(key)
                                    logs[eval_key] = value

                            loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                            learning_rate_scalar = scheduler.get_last_lr()[0]
                            logs["learning_rate"] = learning_rate_scalar
                            logs["loss"] = loss_scalar
                            logging_loss = tr_loss

                            if self.tb_writer:
                                for k, v in logs.items():
                                    self.tb_writer.add_scalar(k, v, global_step)
                            epoch_iterator.write(json.dumps({**logs, **{"step": global_step}}))

                        if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                            # In all cases (even distributed/parallel), self.model is always a reference
                            # to the model we want to save.
                            if hasattr(model, "module"):
                                assert model.module is self.model
                            else:
                                assert model is self.model
                            # Save model checkpoint
                            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
                            self.save_model(output_dir)
                            self._rotate_checkpoints()
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(global_step, tr_loss / global_step)

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        # multi-gpu eval
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model
        model.to(self.args.device)

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", dataloader.batch_size)
        eval_losses: List[float] = []
        preds: np.ndarray = None
        label_ids: np.ndarray = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if self.args.classify_or_insertion == 'classify':
                    padding_max_length = self.model.config.max_position_embeddings
                elif self.args.classify_or_insertion == 'insertion':
                    padding_max_length = self.model.encoder.config.max_position_embeddings
                mode = self.args.classify_or_insertion
                if preds is None:
                    preds = self.padding_ndarray(
                                        ndarray=self.convert_to_np_array(logits, description=mode),
                                        padding_max_length=padding_max_length,
                                        axis=1, 
                                        padding_id=-100
                                        )
                else:
                    tmp_preds = self.padding_ndarray(
                                        ndarray=self.convert_to_np_array(logits, description=mode),
                                        padding_max_length=padding_max_length,
                                        axis=1, 
                                        padding_id=-100
                                        )
                    preds = np.append(preds, tmp_preds, axis=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = self.padding_ndarray(
                                        ndarray=inputs["labels"].detach().cpu().numpy(), 
                                        padding_max_length=padding_max_length,
                                        axis=1, 
                                        padding_id=-100
                                        )
                    else:
                        tmp_label_ids = self.padding_ndarray(
                                        ndarray=inputs["labels"].detach().cpu().numpy(), 
                                        padding_max_length=padding_max_length,
                                        axis=1, 
                                        padding_id=-100
                                        )
                        label_ids = np.append(label_ids, tmp_label_ids, axis=0)
                elif inputs.get("masked_lm_labels") is not None:
                    if label_ids is None:
                        label_ids = self.padding_ndarray(
                                        ndarray=inputs["masked_lm_labels"].detach().cpu().numpy(), 
                                        padding_max_length=padding_max_length,
                                        axis=1, 
                                        padding_id=-100
                                        )
                    else:
                        tmp_label_ids = self.padding_ndarray(
                                        ndarray=inputs["masked_lm_labels"].detach().cpu().numpy(), 
                                        padding_max_length=padding_max_length,
                                        axis=1, 
                                        padding_id=-100
                                        )
                        label_ids = np.append(label_ids, tmp_label_ids, axis=0)
        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(self.args, EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["loss"] = np.mean(eval_losses)
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def padding_ndarray(self, ndarray=None, padding_max_length=512, axis=1, padding_id=-100):
        padding_array_shape = list(ndarray.shape)
        for i, size in enumerate(padding_array_shape):
            if i == axis:
                padding_array_shape[i] = padding_max_length - size
        padding_array = np.zeros(padding_array_shape)
        padding_array.fill(padding_id)
        # print(ndarray.shape, '\n', padding_array.shape)
        result = np.concatenate((ndarray, padding_array), axis=axis)
        return result

    def convert_to_np_array(self, torch_tensor, description=None):
        if description == 'classify':
            return torch_tensor.detach().cpu().numpy()
        elif description == 'insertion':
            return np.argmax(torch_tensor.detach().cpu().numpy(), axis=2)

    def predict(self, test_dataset=None, mode=None) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        assert mode in ['classify', 'insertion']
        test_dataloader = self.get_test_dataloader(test_dataset)
        if mode=='classify':
            return self._inference(test_dataloader, description="Detection")
        elif mode =='insertion':
            return self._inference(test_dataloader, description="Editing")

    def _inference(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:

        assert description in ['Detection', 'Editing']

        # multi-gpu eval
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model
        model.to(self.args.device)

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", dataloader.batch_size)

        if description == 'Detection':
            model.eval()
            input_ids = []
            # input_ids = a list of batches, for each batch is [batch_size, seq_len]
            predictions = []

            for inputs in tqdm(dataloader, desc=description):
                has_labels = any(inputs.get(k) is not None for k in ["labels", "masked_lm_labels"])
                if has_labels:
                    raise ValueError('Labels shouldn\'t be included for inference. ') 

                for k, v in inputs.items():
                    inputs[k] = v.to(self.args.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs[0]
                predictions.append(torch.argmax(logits, dim=2))
                input_ids.append(inputs["input_ids"])
            return predictions, input_ids

        elif description == 'Editing':
            encoder_input_ids, decoder_output_ids = [], []
            # encoder_input_ids = a list of tensors of [seq_len]
            model.eval()

            for inputs in tqdm(dataloader, desc=description):
                has_labels = any(inputs.get(k) is not None for k in ["labels", "masked_lm_labels"])
                if has_labels:
                    raise ValueError('Labels shouldn\'t be included in input for inference. ') 

                for k, v in inputs.items():
                    inputs[k] = v.to(self.args.device)

                for sample_count in range(inputs['input_ids'].shape[0]):
                    temp_inputs = dict()
                    for k, v in inputs.items():
                        temp_inputs[k] = v[sample_count].unsqueeze(0)
                    insertion_position = []
                    for input_ids_position, _ in enumerate(temp_inputs['input_ids'][0]):
                        # find [soe] token position
                        if _ == START_OF_ENTITY_index:
                            temp_inputs['input_ids'][0][input_ids_position] = 103
                            temp_inputs['decoder_input_ids'][0][input_ids_position] = 103
                            insertion_position.append(input_ids_position)
                    for inference_step in range(self.args.max_inference_len):
                        with torch.no_grad():
                            outputs = model(**temp_inputs)
                            logits = outputs[0]
                            temp_inputs['encoder_outputs'] = outputs[1:]
                        preds = np.argmax(logits.detach().cpu().numpy(), axis=2)
                        # preds = [1, seq_len]
                        temp_decoder_input_ids = temp_inputs['decoder_input_ids']
                        del_insertion_position = []
                        for temp_position, _ in enumerate(insertion_position):
                            if preds[0, _] == END_OF_ENTITY_index:
                                del_insertion_position.append(temp_position)
                                continue
                            temp_decoder_input_ids = torch.cat(
                                                                [temp_decoder_input_ids[0, :_+temp_position], torch.tensor([preds[0, _]]).type_as(torch.LongTensor()).to('cuda'), temp_decoder_input_ids[0, _+temp_position:]],
                                                                0
                                                                ).unsqueeze(0)
                        del_insertion_position.sort(reverse=True)
                        for _ in del_insertion_position:
                            del insertion_position[_]
                        temp_inputs['decoder_input_ids'] = temp_decoder_input_ids
                        if len(insertion_position) == 0:
                            break
                        for pp in range(len(insertion_position)):
                            insertion_position[pp] += (pp+1)
                    decoder_output_ids.append(temp_inputs['decoder_input_ids'].squeeze(0))
                    encoder_input_ids.append(temp_inputs['input_ids'].squeeze(0))
            return decoder_output_ids, encoder_input_ids