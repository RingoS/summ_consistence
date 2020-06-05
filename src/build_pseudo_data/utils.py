

third_personal_pronoun_subjective = ['she', 'he'] #0
third_personal_pronoun_objective = ['her', 'him'] #1
third_personal_pronoun_possesive = ['her', 'his'] #2
third_personal_pronoun_possesive_absolute = ['hers', 'his'] #3
third_personal_pronoun_reflexive = ['herself', 'himself'] #4
pronoun_lists = \
    [third_personal_pronoun_subjective, \
        third_personal_pronoun_objective, \
        third_personal_pronoun_possesive, \
        third_personal_pronoun_possesive_absolute, \
        third_personal_pronoun_reflexive]

all_third_pronoun = set(third_personal_pronoun_subjective) \
                    .union(set(third_personal_pronoun_objective)) \
                    .union(set(third_personal_pronoun_possesive)) \
                    .union(set(third_personal_pronoun_possesive_absolute)) \
                    .union(set(third_personal_pronoun_reflexive))

female_pronoun = {'she', 'her', 'hers', 'herself'}
male_pronoun = {'he', 'him', 'his', 'himself'}

number_type = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

name_type = ['PERSON', 'OTHER']

# Stop words
STOP_WORDS = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at
back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by
call can cannot ca could
did do does doing done down due during
each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except
few fifteen fifty first five for former formerly forty four from front full
further
get give go
had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred
i if in indeed into is it its itself
keep
last latter latterly least less
just
made make many may me meanwhile might mine more moreover most mostly move much
must my myself
name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere
of off often on once one only onto or other others otherwise our ours ourselves
out over own
part per perhaps please put
quite
rather re really regarding
same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such
take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two
under until up unless upon us used using
various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would
yet you your yours yourself yourselves
""".split()
)

contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
STOP_WORDS.update(contractions)

for apostrophe in ["‘", "’"]:
    for stopword in contractions:
        STOP_WORDS.add(stopword.replace("'", apostrophe))


#s