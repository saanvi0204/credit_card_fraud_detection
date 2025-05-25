# Credit Card Fraud Detection using Hybrid Autoencoder Model 
### Data preprocessing
• Performed exploratory data analysis (EDA) including distribution plots for Hour, Amount, and log-transformed amount (Log_Amount). <br/>
• Dropped unneeded columns (Time, Amount) and reorganized the dataset.<br/>
• Visualized feature distributions for fraud vs. legitimate transactions using KDE plots.<br/>
• Plotted a correlation heatmap to understand feature relationships.<br/>
• Trained a Random Forest model to identify and rank feature importances.<br/>
• Used t-SNE for 2D visualization of a balanced subset (fraud and undersampled legitimate data) to analyze class separability.<br/>
### Implementing autoencoder
• Split data into features (X) and labels (y), and scaled features using MinMaxScaler.<br/>
• Split the dataset into train/test and further into train/validation using only legitimate (non-fraud) data for training.<br/>
• Defined a deep autoencoder architecture with Dense + BatchNorm + ReLU + Dropout layers; L2 regularization and dropout to avoid overfitting.<br/>
• Compiled the model with MSE loss and Adam optimizer.<br/>
• Trained the autoencoder with callbacks EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau.<br/>
• Applied t-SNE on latent features and plotted a 2D scatter plot showing separation between fraud and legitimate classes.<br/>
### Comparing ensemble methods for classification
• Split data into train and test sets using stratified sampling.<br/>
• Combined latent features + reconstruction error to form enriched features for classification.<br/>
• Used RandomizedSearchCV with StratifiedKFold CV to tune each model (Random forest, Adaboost, Gradient boosting and LightGBM) on the enriched feature set.<br/>
• Evaluated each model on test data using Accuracy, TPR, TNR, F1 Score, MCC (mainly TPR and MCC).<br/>
### Fine-tuning classification model
• Set up 5-fold StratifiedKFold cross-validation.<br/>
• Initialized a CSV logger to record experiment results.<br/>
• Split training data further for calibration.<br/>
• Built a StackingClassifier with tuned LGBM and RF, using logistic regression as meta-learner.<br/>
• Fitted the stack model on training data.<br/>
Applied Platt scaling (logistic regression) on predicted probabilities from calibration set.<br/>
Tuned decision threshold using MCC on calibration set.<br/>
Calibrated test probabilities and applied optimal threshold.<br/>



