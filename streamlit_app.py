import streamlit as st
import pandas as pd
import numpy as np
#to divide the sata into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #converts lavels into numbers for ML

#start building
st.set_page_config(page_title="Mushroom Classifier", layout="centered")
st.markdown(
    """
    <style>
        /* Page background gradient */
        body {
            background: linear-gradient(to bottom, #2d6a4f, #d8f3dc);
            color: #1b4332;
        }

        .stApp {
            background: linear-gradient(to bottom, #2d6a4f, #d8f3dc);
            color: #1b4332;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Title color */
        h1 {
            color: #1b4332;
        }

        /* Buttons */
        .stButton > button {
            background-color: #40916c;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #006400;
        }

        /* Dropdowns */
        .css-1cpxqw2, .css-1r6slb0 {
            background-color: #d8f3dc !important;
            color: #1b4332 !important;
            border-radius: 8px;
        }

        .css-1d391kg, .css-14el2xx {
            color: #1b4332 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🍄Mushroom Safety Checker🍄")
st.write("Welcome to the Mushroom Safety Checker app! Don't know if it's safe to eat? Let us help you! How does your mushroom look? 👀")

#load your dataset
df = pd.read_excel("mushrooms_dataset.xlsx")
# Replace '?' with 'missing' so LabelEncoder can handle it properly
df.replace('?', 'missing', inplace=True)
#print(df.head()) #check if the dataset has been loaded correctly

#data cleaning part
#check for missing values
#st.write("Missing values in each column: ")
#st.write(df.isnull().sum())
#there are no mmissing values

#check for duplicates
#st.write("Number of duplicate rows: ", df.duplicated().sum())
#df.drop_duplicates(inplace=True)

#confirm data types of each column
#st.write("Data types of each column: ")
#st.write(df.dtypes)

#data encoding
#this is because ML models only work with numerical data. Our data set has labels that need to be converted to numbers first
#Encoding approach: Label Encoding - assigns a unique number to each category in each column
#first, create a copy dataset to avoid overwriting the original
df_encoded = df.copy()
#create a dictionaty to store encoders (for decoding later if needed)
label_encoders = {}

#loop through  all the columns
df_encoded = df.copy()
label_encoders = {}

for column in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le
#preview encoded data
#st.write("Encoded dataset: ")
#st.dataframe(df_encoded.head())

#train-test split
#features (X) = all columns except target - creates a dataset with only the features
X = df_encoded.drop('class', axis = 1)

#Target(y) = whether the mushroom is edible or poisonous - this is your target variable. edible(1) or poisonous (0)
y = df_encoded['class']

#split the data - 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.2, random_state = 42
)

#print the dataset shapes to confirm
#st.write("Training set shape: ", X_train.shape)
#st.write("Testing set shape: ", X_test.shape)

#6,499 mushrooms for training the model
#1,625 mushrooms to test how well your model performs
#22 features (like cap color, odor, habitat, etc.)

#Model training and Evaluation
#choose an algorithm - Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Create the model
model = RandomForestClassifier(random_state = 42) #Initializes the model, accuracy = 1.0
#train the model
model.fit(X_train, y_train)#trains the model using the training data
#Predict on the test set
y_pred = model.predict(X_test)#makes predictions on test data
#Evaluate performance
accuracy = accuracy_score(y_test, y_pred)#checks how many predictions were correct

#Show results in Streamlit
#st.subheader("Model evaluation")
#st.write("Accuracy: ", accuracy)
#st.write("Classification Report: ")
#st.text(classification_report(y_test, y_pred)) #shows precision, recall and f-1 score
#st.write("Confusion Matrix: ")
#st.write(confusion_matrix(y_test,y_pred))#shows how many times the model got edible/poisonous right or wrong

#model 2: Logistic regression 0.9464615384615385~accuracy
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#import streamlit as st

# Train the model
#log_model = LogisticRegression(max_iter=200, random_state=42)
#log_model.fit(X_train, y_train)

# Make predictions
#y_pred = log_model.predict(X_test)

# Evaluate the model
#accuracy = accuracy_score(y_test, y_pred)

# Streamlit output (formatted same as your previous model)
#st.subheader("Model evaluation")
#st.write("Accuracy: ", accuracy)
#st.write("Classification Report: ")
#st.text(classification_report(y_test, y_pred))  # shows precision, recall and f1-score
#st.write("Confusion Matrix: ")
#st.write(confusion_matrix(y_test, y_pred))  # shows how many times the model got edible/poisonous right or wrong

#Naive Bayes model  ~ 0.9218461538461539
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#import streamlit as st

# Train the Naive Bayes model 
#nb_model = GaussianNB()
#nb_model.fit(X_train, y_train)

# Make predictions
#y_pred = nb_model.predict(X_test)

# Evaluate the model
#accuracy = accuracy_score(y_test, y_pred)

# Streamlit output
#st.subheader("Model evaluation")
#st.write("Accuracy: ", accuracy)
#st.write("Classification Report: ")
#st.text(classification_report(y_test, y_pred))  # shows precision, recall and f1-score
#st.write("Confusion Matrix: ")
#st.write(confusion_matrix(y_test, y_pred))  # shows how many times the model got edible/poisonous right or wrong

st.subheader("Enter Mushroom Features: ")

def create_selectbox(label, options_dict, column_name):
    encoder_classes = set(label_encoders[column_name].classes_)
    valid_options = {k: v for k, v in options_dict.items() if v in encoder_classes}

    if not valid_options:
        st.error(f"No valid options for '{column_name}' — check your dataset.")
        st.stop()

    choice = st.selectbox(label, list(valid_options.keys()))
    value = valid_options[choice]
    return label_encoders[column_name].transform([value])[0]


# User-friendly dropdowns with internal codes
cap_shape = create_selectbox("Cap Shape", {
    "Bell": "b", "Conical": "c", "Convex": "x", "Flat": "f", "Knobbed": "k", "Sunken": "s"
}, "cap-shape")

cap_surface = create_selectbox("Cap Surface", {
    "Fibrous": "f", "Grooves": "g", "Scaly": "y", "Smooth": "s"
}, "cap-surface")

cap_color = create_selectbox("Cap Color", {
    "Brown": "n", "Buff": "b", "Cinnamon": "c", "Gray": "g", "Green": "r", "Pink": "p",
    "Purple": "u", "Red": "e", "White": "w", "Yellow": "y"
}, "cap-color")

bruises = create_selectbox("Bruises", {
    "Yes": "t", "No": "f"
}, "bruises")

odor = create_selectbox("Odor", {
    "Almond": "a", "Anise": "l", "Creosote": "c", "Fishy": "y", "Foul": "f",
    "Musty": "m", "None": "n", "Pungent": "p", "Spicy": "s"
}, "odor")

gill_attachment = create_selectbox("Gill Attachment", {
    "Attached": "a", "Descending": "d", "Free": "f", "Notched": "n"
}, "gill-attachment")

gill_spacing = create_selectbox("Gill Spacing", {
    "Close": "c", "Crowded": "w", "Distant": "d"
}, "gill-spacing")

gill_size = create_selectbox("Gill Size", {
    "Broad": "b", "Narrow": "n"
}, "gill-size")

gill_color = create_selectbox("Gill Color", {
    "Black": "k", "Brown": "n", "Buff": "b", "Chocolate": "h", "Gray": "g", "Green": "r",
    "Orange": "o", "Pink": "p", "Purple": "u", "Red": "e", "White": "w", "Yellow": "y"
}, "gill-color")

stalk_shape = create_selectbox("Stalk Shape", {
    "Enlarging": "e", "Tapering": "t"
}, "stalk-shape")

stalk_root = create_selectbox("Stalk Root", {
    "Bulbous": "b", "Club": "c", "Cup": "u", "Equal": "e", "Rhizomorphs": "z",
    "Rooted": "r", "Missing": "missing"
}, "stalk-root")

stalk_surface_above_ring = create_selectbox("Stalk Surface Above Ring", {
    "Fibrous": "f", "Scaly": "y", "Silky": "k", "Smooth": "s"
}, "stalk-surface-above-ring")

stalk_surface_below_ring = create_selectbox("Stalk Surface Below Ring", {
    "Fibrous": "f", "Scaly": "y", "Silky": "k", "Smooth": "s"
}, "stalk-surface-below-ring")

stalk_color_above_ring = create_selectbox("Stalk Color Above Ring", {
    "Brown": "n", "Buff": "b", "Cinnamon": "c", "Gray": "g", "Orange": "o",
    "Pink": "p", "Red": "e", "White": "w", "Yellow": "y"
}, "stalk-color-above-ring")

stalk_color_below_ring = create_selectbox("Stalk Color Below Ring", {
    "Brown": "n", "Buff": "b", "Cinnamon": "c", "Gray": "g", "Orange": "o",
    "Pink": "p", "Red": "e", "White": "w", "Yellow": "y"
}, "stalk-color-below-ring")

veil_type = create_selectbox("Veil Type", {
    "Partial": "p", "Universal": "u"
}, "veil-type")

veil_color = create_selectbox("Veil Color", {
    "Brown": "n", "Orange": "o", "White": "w", "Yellow": "y"
}, "veil-color")

ring_number = create_selectbox("Ring Number", {
    "None": "n", "One": "o", "Two": "t"
}, "ring-number")

ring_type = create_selectbox("Ring Type", {
    "Cobwebby": "c", "Evanescent": "e", "Flaring": "f", "Large": "l", "None": "n",
    "Pendant": "p", "Sheathing": "s", "Zone": "z"
}, "ring-type")

spore_print_color = create_selectbox("Spore Print Color", {
    "Black": "k", "Brown": "n", "Buff": "b", "Chocolate": "h", "Green": "r",
    "Orange": "o", "Purple": "u", "White": "w", "Yellow": "y"
}, "spore-print-color")

population = create_selectbox("Population", {
    "Abundant": "a", "Clustered": "c", "Numerous": "n", "Scattered": "s",
    "Several": "v", "Solitary": "y"
}, "population")

habitat = create_selectbox("Habitat", {
    "Grasses": "g", "Leaves": "l", "Meadows": "m", "Paths": "p",
    "Urban": "u", "Waste": "w", "Woods": "d"
}, "habitat") 

#user dict
user_input_dict = {
    'cap-shape': cap_shape,
    'cap-surface': cap_surface,
    'cap-color': cap_color,
    'bruises': bruises,
    'odor': odor,
    'gill-attachment': gill_attachment,
    'gill-spacing': gill_spacing,
    'gill-size': gill_size,
    'gill-color': gill_color,
    'stalk-shape': stalk_shape,
    'stalk-root': stalk_root,
    'stalk-surface-above-ring': stalk_surface_above_ring,
    'stalk-surface-below-ring': stalk_surface_below_ring,
    'stalk-color-above-ring': stalk_color_above_ring,
    'stalk-color-below-ring': stalk_color_below_ring,
    'veil-type': veil_type,
    'veil-color': veil_color,
    'ring-number': ring_number,
    'ring-type': ring_type,
    'spore-print-color': spore_print_color,
    'population': population,
    'habitat': habitat
}

#create a dataset
user_input_df = pd.DataFrame([user_input_dict])

if st.button("Check if Mushroom is Safe"):
    prediction = model.predict(user_input_df)[0]
    decoded_prediction = label_encoders['class'].inverse_transform([prediction])[0]

    st.subheader("Prediction Result")
    if decoded_prediction == 'e':
        st.success("This mushroom is **Edible** ✅")
    else:
        st.error("This mushroom is **Poisonous** ⚠️")
        
