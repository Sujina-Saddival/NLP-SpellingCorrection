# NLP-SpellingCorrection
#Run spellcorrect

## python3 spellcorrect_starter.py 1 dev.txt > corpus.keys.txt

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
Num correct:  85
Total:  421
Accuracy: 0.202