import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Load Dataset
data = pd.read_csv("Blockchain103.csv")

# Step 3: Data Preprocessing
features = [
    "BlockHeight",
    "UnixTimestamp",
    "TxnFee(ETH)",
    "TxnFee (Binary)",
    "Block Generation Rate",
    "Stake Reward",
    "Coin Stake",
    "Stake Distribution Rate",
    "Txnsize",
    "Coin Days",
    "Coin Age",
    "Block Density (%)",
    "Block Score",
    "Coin Day Weight",
    "Transaction Velocity",
    "Node Efficiency",
    "Network Latency",
]

X = data[features]

# Convert 'Node Uptime' to numeric (assuming you want to use this feature)
data["Node Uptime"] = pd.to_timedelta(data["Node Uptime"]).dt.total_seconds()

# Drop rows with NaN values, if any
X = X.dropna()

y = data["Node Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the Model
svm_model = SVC(kernel="linear", C=0.1, gamma="scale")
svm_model.fit(X_train_scaled, y_train)

# Step 5: Make Predictions
data_scaled = scaler.transform(X)
data["Predicted_Label"] = svm_model.predict(data_scaled)

# Step 6: Identify Potentially Malicious Nodes
potentially_malicious_nodes = data[data["Predicted_Label"] == 1]

# Display Results
print("\n Step 1-3: Data Loaded and Preprocessed\n")
print("Step 4: Model Training Completed\n")
print("Step 5: Predictions Made\n")
print("Step 6: Potentially Malicious Nodes Identified:\n")
print(potentially_malicious_nodes, "\n")
