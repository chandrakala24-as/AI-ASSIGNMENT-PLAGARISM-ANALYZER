from algorithms.tfidf_cosine import calculate_tfidf_similarity_batch
from algorithms.ngram_matching import calculate_ngram_similarity
from algorithms.winnowing import WinnowingMatcher
from algorithms.crnn_inference import calculate_crnn_similarity_batch

target = "This is a sample document to test the plagiarism detection algorithms. It contains enough words to be meaningful."
peer = "This is a sample document to test the plagiarism detection algorithms. It contains enough words to be meaningful."

print("TFIDF:", calculate_tfidf_similarity_batch(target, [peer]))
print("Ngram:", calculate_ngram_similarity(target, peer))
wm = WinnowingMatcher(k=12, w=4)
print("Winnow:", wm.calculate_similarity(target, peer)[0])
print("CRNN:", calculate_crnn_similarity_batch(target, [peer]))
