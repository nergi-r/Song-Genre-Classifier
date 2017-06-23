# Song-Genre-Classifier

An application to predict Indonesiang songs genre based on its lyric.

How it works:
1. Convert the given lyric into vector space using CountVectorizer
2. Convert into Tf-Idf matrix using TfidfTransformer
3. Apply Latent Semantic Analysis (LSA) to the matrix by using Non-negative Matrix Factorization technique.
4. Use trained SVM Classifier to predict the genre.

Training set:

Manually searched from Indonesian songs lyric websites
