# Clinical Appointment No-Show Prediction (Milestone 1)

This project implements a traditional machine learning pipeline to predict whether a patient will miss a scheduled clinical appointment (no-show). The goal is to help healthcare providers identify high-risk appointments and take proactive measures to reduce no-shows.

---

## 🔍 Problem Statement

Missed clinical appointments lead to wasted resources and reduced healthcare efficiency.  
This project aims to build a supervised ML model that predicts the probability of a patient not showing up for an appointment based on demographic and appointment-related features.

---

## 📁 Dataset

**Source:** Kaggle – Medical Appointment No-Shows  
**Target Variable:** `No-show`

- `1` = Patient did not show up
- `0` = Patient showed up

**Features Used (examples):**

- Age
- Gender
- Neighbourhood
- SMS_received
- ScheduledDay, AppointmentDay
- Engineered Feature: `WaitingDays`

---

## 🧪 Exploratory Data Analysis (EDA)

- Analyzed class distribution of no-shows vs shows
- Studied impact of SMS reminders on attendance
- Explored relationship between waiting time and no-show probability

---

## ⚙️ Methodology

The problem is framed as a **binary classification** task.

**Models Implemented:**

- Logistic Regression (baseline, interpretable model)
- Decision Tree (to capture non-linear patterns)

**Preprocessing Steps:**

- Handling missing values
- Feature engineering (`WaitingDays`)
- One-hot encoding for categorical variables
- Standardization of numerical features
- Stratified train-test split

---

## 📊 Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The model with better balanced precision–recall performance was selected as the final model for inference.
