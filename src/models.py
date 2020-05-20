import torch
import torch.nn as nn
import transformers.modeling_bert as modeling_bert
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers import (
                        AutoModel, 
                        AutoConfig, 
                        AutoModelWithLMHead, 
                        EncoderDecoderConfig, 
                        EncoderDecoderModel, 
                        PreTrainedModel, 
                        PretrainedConfig
                        )
from typing import Optional

@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, modeling_bert.BERT_START_DOCSTRING)
class BertForTextEditing(modeling_bert.BertPreTrainedModel):
    '''
    This model is modified on the basis of BertForMaskedLM and BertForTokenClassification
    '''
    def __init__(self, config=None, model_label=None, model_gen=None):
        super().__init__(config)

        if config:
            self.config = config
            self.share_bert_param = config.share_bert_param
            if not(model_label or model_gen):
                if config.share_bert_param:
                    self.bert = modeling_bert.BertModel(config)
                else:
                    self.bert_label = modeling_bert.BertModel(config)
                    self.bert_gen = modeling_bert.BertModel(config)
        elif model_label and model_gen:
            self.bert_label = model_label
            self.bert_gen = model_gen
        elif model_label and not model_gen:
            self.bert = model_label

        # sequence labeling
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_label = nn.Linear(config.hidden_size, config.num_labels)

        # generation(masked lm)
        self.cls_gen = modeling_bert.BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls_gen.predictions.decoder


    @add_start_docstrings_to_callable(modeling_bert.BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        labels=None,
        description='gen',
    ):
        r"""
        # for sequence labeling task
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the token classification loss.
                Indices should be in ``[0, ..., config.num_labels - 1]``.
                
        # for generation(masked lm) task
            masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the left-to-right language modeling loss (next word prediction).
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``

    Returns:
        # for sequence labeling task
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
                Classification loss.
            scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
                Classification scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

        # for generation(masked lm) task
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Masked language modeling loss.
            ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                    Next token prediction loss.
            prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

            Examples::
                # for sequence labeling task
                    from transformers import BertTokenizer, BertForTokenClassification
                    import torch

                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    model = BertForTokenClassification.from_pretrained('bert-base-uncased')

                    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
                    labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
                    outputs = model(input_ids, labels=labels)

                    loss, scores = outputs[:2]

                # for generation(masked lm) task
                    from transformers import BertTokenizer, BertForMaskedLM
                    import torch

                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

                    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
                    outputs = model(input_ids, masked_lm_labels=input_ids)

                    loss, prediction_scores = outputs[:2]

        """

        if description == 'label':
            if self.share_bert_param:
                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                )
            else:
                outputs = self.bert_label(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                )

            sequence_output = outputs[0]

            sequence_output = self.dropout(sequence_output)
            logits = self.classifier_label(sequence_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs  # (loss), scores, (hidden_states), (attentions)

        elif description == 'gen':
            if self.share_bert_param:
                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                outputs = self.bert_gen(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

            sequence_output = outputs[0]
            prediction_scores = self.cls_gen(sequence_output)

            outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

            # Although this may seem awkward, BertForMaskedLM supports two scenarios:
            # 1. If a tensor that contains the indices of masked labels is provided,
            #    the cross-entropy is the MLM cross-entropy that measures the likelihood
            #    of predictions for masked words.
            # 2. If `lm_labels` is provided we are in a causal scenario where we
            #    try to predict the next token for each input in the decoder.
            if masked_lm_labels is not None:
                loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs = (masked_lm_loss,) + outputs

            if lm_labels is not None:
                # we are doing next-token prediction; shift prediction scores and input ids by one
                prediction_scores = prediction_scores[:, :-1, :].contiguous()
                lm_labels = lm_labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
                outputs = (ltr_lm_loss,) + outputs

            return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

        else:
            raise ValueError('description {} is not supported by the model'.format(description))

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if model is does not use a causal mask then add a dummy token
        if self.config.is_decoder is False:
            assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1
            )

            dummy_token = torch.full(
                (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
            )
            input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    @classmethod
    def from_pretrained_multi(cls, 
                            config = None,
                            label_pretrained_model_name_or_path: str = None,
                            gen_pretrained_model_name_or_path: str = None,
                            config_label = None,
                            config_gen = None,
                            cache_dir = None,
                            # *model_args,
                            # **kwargs
                            ):
        if config.share_bert_param:
            bert = modeling_bert.BertModel.from_pretrained(
                label_pretrained_model_name_or_path,
                cache_dir=cache_dir,
                )
            return cls(config=config, model_label=bert)
        else:
            bert_label = modeling_bert.BertModel.from_pretrained(
                label_pretrained_model_name_or_path,
                cache_dir=cache_dir,
                )
            bert_gen = modeling_bert.BertModel.from_pretrained(
                gen_pretrained_model_name_or_path,
                cache_dir=cache_dir,
                )
            return cls(config=config, model_label=bert_label, model_gen=bert_gen)

# class BertForARInsertion(modeling_bert.BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = modeling_bert.BertModel(config)
#         self.cls = modeling_bert.BertOnlyMLMHead(config)

#         self.init_weights()

#     def make_tgt_mask(self, tgt):
        
class EncoderDecoderInsertionModel(PreTrainedModel):
    r"""
        :class:`~transformers.EncoderDecoder` is a generic model class that will be
        instantiated as a transformer architecture with one of the base model
        classes of the library as encoder and another one as
        decoder when created with the `AutoModel.from_pretrained(pretrained_model_name_or_path)`
        class method for the encoder and `AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path)` class method for the decoder.
    """
    config_class = EncoderDecoderConfig

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        super().__init__(config)

        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from transformers import AutoModelWithLMHead

            decoder = AutoModelWithLMHead.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder
        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"

    def tie_weights(self):
        # for now no weights tying in encoder-decoder
        pass


    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()


    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()


    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r""" Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).
        To train the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

        Examples::

            from tranformers import EncoderDecoder

            model = EncoderDecoder.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert
        """

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)
        encoder.config.is_decoder = False

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            from transformers import AutoModelWithLMHead

            decoder = BertForMaskedLM_modified.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        decoder.config.is_decoder = True

        model = cls(encoder=encoder, decoder=decoder)

        return model


    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        decoder_inputs_embeds=None,
        masked_lm_labels=None,
        lm_labels=None,
        **kwargs,
    ):

        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the encoder.
                Indices can be obtained using :class:`transformers.PretrainedTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Mask to avoid performing attention on padding token indices for the encoder.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules for the encoder.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
                Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
                `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
                Used in the cross-attention of the decoder.
            decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
                Provide for sequence to sequence training to the decoder.
                Indices can be obtained using :class:`transformers.PretrainedTokenizer`.
                See :func:`transformers.PreTrainedTokenizer.encode` and
                :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
                Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            decoder_head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
                Mask to nullify selected heads of the self-attention modules for the decoder.
                Mask values selected in ``[0, 1]``:
                ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
            decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
                Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss for the decoder.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                Labels for computing the left-to-right language modeling loss (next word prediction) for the decoder.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            kwargs: (`optional`) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:
                - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
                - With a `decoder_` prefix which will be input as `**decoder_kwargs` for the decoder forward function.

        Examples::

            from transformers import EncoderDecoderModel, BertTokenizer
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert

            # forward
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            # training
            loss, outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)[:2]

            # generation
            generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

        """

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                **kwargs_encoder,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            lm_labels=lm_labels,
            masked_lm_labels=masked_lm_labels,
            **kwargs_decoder,
        )

        return decoder_outputs + encoder_outputs


    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step
        if type(past) is tuple:
            encoder_outputs = past
        else:
            encoder_outputs = (past,)

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)

        return {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
        }

    def _reorder_cache(self, past, beam_idx):
        # as a default encoder-decoder models do not re-order the past.
        # TODO(PVP): might have to be updated, e.g. if GPT2 is to be used as a decoder
        return past


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, modeling_bert.BERT_START_DOCSTRING)
class BertForMaskedLM_modified(modeling_bert.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = modeling_bert.BertModel(config)
        self.cls = BertOnlyMLMHead_modified(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(modeling_bert.BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import BertTokenizer, BertForMaskedLM
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss, prediction_scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if model is does not use a causal mask then add a dummy token
        if self.config.is_decoder is False:
            assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1
            )

            dummy_token = torch.full(
                (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
            )
            input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class BertLMPredictionHead_modified(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = modeling_bert.BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size+2, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size+2))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead_modified(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead_modified(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores