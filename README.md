# NLP-SpellingCorrection
#Run spellcorrect

## python3 spellcorrect_starter.py 1 dev.txt > corpus.keys.txt
## python3 spellcorrect_starter.py 2 dev.txt > corpus.keys.txt
## python3 spellcorrect_starter.py interp dev.txt > corpus.keys.txt


## python3 score.py corpus.keys.txt dev.keys.txt


sujinasaddival origin-NLP-a2 $ python3 score.py corpus.keys.txt dev.keys.txt
Num correct:  76
Total:  421
Accuracy: 0.181

Unigram
sujinasaddival origin-NLP-a2 $ python3 spellcorrect_starter.py 1 dev.txt > corpus.keys.txt
sujinasaddival origin-NLP-a2 $ python3 score.py corpus.keys.txt dev.keys.txt
Num correct:  164
Total:  421
Accuracy: 0.39

Bigram
Num correct:  236
Total:  421
Accuracy: 0.561

return Decimal(((1 - lamda) * bigram_prob_1) + (lamda * unigram_prod_1) * ((1 - lamda) * bigram_prob_2) + (lamda * unigram_prod_2))
Without bracket
Inter
Num correct:  210
Total:  421
Accuracy: 0.499

return Decimal((((1 - lamda) * bigram_prob_1) + (lamda * unigram_prod_1)) * (((1 - lamda) * bigram_prob_2) + (lamda * unigram_prod_2)))
With bracket
Num correct:  195
Total:  421
Accuracy: 0.463

Without smoothing -
With brackets
Num correct:  229
Total:  421
Accuracy: 0.544

Lamda - 0.8
Num correct:  223
Total:  421
Accuracy: 0.53

lamda - 0.2
Num correct:  237
Total:  421
Accuracy: 0.563

lamda = 0.00001
Num correct:  244
Total:  421
Accuracy: 0.58