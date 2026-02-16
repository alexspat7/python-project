Serial Killer Identification Project (Machine Learning)
Αυτό το repository περιέχει μια σειρά από Python scripts που αναπτύχθηκαν για την ανάλυση εγκληματολογικών δεδομένων και την ταυτοποίηση δραστών (Serial Killers) χρησιμοποιώντας διάφορες τεχνικές Μηχανικής Μάθησης (Machine Learning) και Στατιστικής Ανάλυσης.

Το project ακολουθεί μια σταδιακή προσέγγιση, ξεκινώντας από την Εξερευνητική Ανάλυση Δεδομένων (EDA) και καταλήγοντας σε σύνθετα μοντέλα όπως Νευρωνικά Δίκτυα και Clustering.

Δομή του Project
Τα αρχεία κώδικα είναι χωρισμένα σε βήματα (Questions Q1-Q8), όπου το καθένα εκτελεί μια συγκεκριμένη εργασία ανάλυσης ή μοντελοποίησης.

Περιγραφή Αρχείων
Q1.py - Exploratory Data Analysis (EDA) & GMM

Φόρτωση δεδομένων και δημιουργία ιστογραμμάτων για βασικά χαρακτηριστικά (hour, age, location).

Εφαρμογή Gaussian Mixture Models (GMM) για την ανάλυση της κατανομής των ωρών των εγκλημάτων.

Output: q1_histograms.png.

Q2.py - Maximum Likelihood Estimation (MLE)

Υπολογισμός μέσων τιμών και πινάκων συνδιακύμανσης (Covariance Matrices) για κάθε δολοφόνο ξεχωριστά.

Οπτικοποίηση με Heatmaps και Ελλείψεις Αβεβαιότητας (Confidence Ellipses).

Output: q2_heatmaps.png.

Q3.py - Naive Bayes Classifier

Υλοποίηση ενός προσαρμοσμένου (custom) ταξινομητή Bayes βασισμένου στις MLE παραμέτρους που υπολογίστηκαν προηγουμένως.

Χρήση multivariate_normal για τον υπολογισμό πιθανοτήτων.

Q4.py - Ridge Classifier

Προεπεξεργασία δεδομένων (StandardScaler, OneHotEncoder) μέσω Scikit-Learn Pipeline.

Εκπαίδευση γραμμικού μοντέλου Ridge Regression για ταξινόμηση.

Q5.py - Support Vector Machines (SVM)

Εκπαίδευση μοντέλου SVM με πυρήνα RBF (Radial Basis Function).

Οπτικοποίηση του Confusion Matrix και των Support Vectors μέσω PCA (2D προβολή).

Output: q5_confusion_matrix.png.

Q6.py - Neural Networks (MLP)

Χρήση Πολυεπίπεδου Νευρωνικού Δικτύου (Multi-Layer Perceptron - MLP) με κρυφά επίπεδα (64, 32).

Υπολογισμός και ανάλυση της σημαντικότητας των χαρακτηριστικών (Feature Importance) μέσω μετάθεσης (permutation).

Q7.py - Principal Component Analysis (PCA)

Εφαρμογή PCA για μείωση διαστάσεων.

Δημιουργία Scree Plot για την αξιολόγηση της διακύμανσης που εξηγείται από κάθε συνιστώσα.

Output: q7_scree_plot.png.

Q8.py - Clustering (K-Means)

Μη επιβλεπόμενη μάθηση (Unsupervised Learning) με χρήση K-Means Clustering πάνω στα δεδομένα μειωμένων διαστάσεων (PCA).

Αντιστοίχιση των clusters στους δολοφόνους βάσει πλειοψηφίας (voting).

submission.csv

Το τελικό αρχείο αποτελεσμάτων που περιέχει τις προβλέψεις και τις πιθανότητες για κάθε περιστατικό, έτοιμο για υποβολή/αξιολόγηση.

Απαιτήσεις Συστήματος (Requirements)
Για την εκτέλεση του κώδικα απαιτείται Python 3.x και οι παρακάτω βιβλιοθήκες:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
Οδηγίες Εκτέλεσης
Βεβαιωθείτε ότι το αρχείο δεδομένων crimes.csv βρίσκεται στον ίδιο φάκελο με τα scripts.

Μπορείτε να τρέξετε κάθε script ξεχωριστά ανάλογα με το ερώτημα που θέλετε να εξετάσετε:

Bash
python Q1.py  # Για EDA και γραφήματα
python Q5.py  # Για εκπαίδευση SVM
python Q8.py  # Για Clustering
Δεδομένα (Dataset)
Το dataset (crimes.csv) περιλαμβάνει πληροφορίες για σκηνές εγκλήματος με χαρακτηριστικά όπως:

Χωρικά: latitude, longitude, dist_precinct_km.

Χρονικά: hour_float.

Περιβαλλοντικά: temp_c, humidity, weather.

Θύμα: victim_age, vic_gender.

Στόχος: killer_id (η ετικέτα που προσπαθούμε να προβλέψουμε).

Αποτελέσματα
Τα scripts παράγουν αυτόματα αρχεία εικόνας (.png) με γραφικές παραστάσεις για την κατανόηση των δεδομένων και την αξιολόγηση των μοντέλων, καθώς και εκτυπώσεις στην κονσόλα με την ακρίβεια (accuracy) του κάθε μοντέλου.
