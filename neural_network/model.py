#Need to install rdkit onto machine for code to work 
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')
#conda install -c rdkit rdkit -y

#installing libraries 
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

#Reading dataset
comp = pd.read_csv("raw_esol.csv")
comp
comp.info()

#Converting molecules from dataset columns (SMILES) into rdkit object
from rdkit import Chem
mol=[Chem.MolFromSmiles(drug) for drug in comp.SMILES]
print(len(mol))
mol

#Using rdkit to extract molecule data to set parameters - cLogP, Molecular weight, Rotatable bonds, H-bond donor count, H-bond acceptor count, Polar surface area
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
def parameters(smiles, verbose=False):
    base_data = []
    for s in smiles:
        mol=Chem.MolFromSmiles(s)
        base_data.append(mol)

    data = np.arange(1,1)
    r = 0
    for m in base_data:
        Mol_LogP = Descriptors.MolLogP(m)
        Mol_Weight = Descriptors.MolWt(m)
        Rotable_Bonds = Descriptors.NumRotatableBonds(m)
        H_bond_donor = Lipinski.NumHDonors(m)
        H_bond_acceptor = Lipinski.NumHAcceptors(m)
        Polar_Surface_Area = Descriptors.TPSA(m)

        row_data = np.array([Mol_LogP,
                             Mol_Weight,
                             Rotable_Bonds,
                             H_bond_donor,
                             H_bond_acceptor,
                             Polar_Surface_Area])
        if (r==0):
            data = row_data
        else:
            data = np.vstack([data, row_data])
        r = r + 1

    column_names = ["cLogP", "Molecular weight", "Rotatable bonds", "H-bond donor count", "H-bond acceptor count", "Polar surface area"]
    df = pd.DataFrame(data=data, columns=column_names)
    return df

descriptors = parameters(comp.SMILES)
descriptors

#Extracting aromatic propertions 
def Aro_Atoms(mol):
    aro_atoms = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol.GetNumAtoms())]
    
    a_atoms = []
    for a in aro_atoms:
        if a == True:
            a_atoms.append(a)
    sum_a_atoms = sum(a_atoms)
    return sum_a_atoms

aromatic_proportion = [Aro_Atoms(element)/Descriptors.HeavyAtomCount(element) for element in mol]
aromatic_proportion = pd.DataFrame(aromatic_proportion, columns=['Aromatic proportion'])
aromatic_proportion

#Extracting non carbon propertions
non_carbon_proportion = [Lipinski.NumHeteroatoms(element)/Lipinski.HeavyAtomCount(element) for element in mol]
non_carbon_proportion = pd.DataFrame(non_carbon_proportion, columns=['Non carbon proportion'])
non_carbon_proportion

#X and Y matrix 
X = pd.concat([descriptors, aromatic_proportion, non_carbon_proportion], axis=1)
Y = comp.iloc[:,1]

#Splitting data into training and test (20%) data sets
# Change test data set size to 10%,15% ,20% and 20% and compare rmse and r^2 values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train_shape = X_train.shape
X_test_shape = X_test.shape
Y_train_shape = Y_train.shape
Y_test_shape = Y_test.shape
print(X_train_shape)
print(X_test_shape)
print(Y_train_shape)
print(Y_test_shape)

#Testing first with linear regression - and applying model onto training set
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, Y_train)
linear_Y_train_pred = linear.predict(X_train)
linear_Y_train_pred

print('Coefficients:', linear.coef_)
print('Intercept:', linear.intercept_)

#Calculating RMSE (Root mean squared error - which tells you how far the regression line data points are away from actual values)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_linear_Y_train_pred = sqrt(mean_squared_error(Y_train, linear_Y_train_pred))
rmse_linear_Y_train_pred
print(' Training rmse: ' + str(rmse_linear_Y_train_pred))

#Calculating R^2 Score 
from sklearn.metrics import r2_score
linear_Y_train_score = r2_score(Y_train, linear_Y_train_pred)
linear_Y_train_score
print('Training r^2:'+ str(linear_Y_train_score))

#Plotting linear regression on graph 
plt.figure(figsize=(5,5))
plt.scatter(Y_train, linear_Y_train_pred, c='b')
z = np.polyfit(Y_train, linear_Y_train_pred, 1)
p = np.poly1d(z)
plt.plot(Y_train, p(Y_train), 'r-')
plt.xlabel('Predicted molecule solubility')
plt.ylabel('Measured molecule solubility')
plt.title("Predicted molecule log(solubility) trained against measured molecule log(solubility)")
plt.show()

#Applying linear regression model to testing data set 
linear_Y_test_pred = linear.predict(X_test)
linear_Y_test_pred

print('Coefficients:', linear.coef_)
print('Intercept:', linear.intercept_)

#Calculating RMSE
rmse_linear_Y_test_pred = sqrt(mean_squared_error(Y_test, linear_Y_test_pred))
rmse_linear_Y_test_pred
print('Testing rmse: ' + str(rmse_linear_Y_test_pred))

#Calculating R^2 Score 
linear_Y_test_score = r2_score(Y_test, linear_Y_test_pred)
linear_Y_test_score
print('Testing r^2:'+ str(linear_Y_test_score))

#Plotting testing data 
plt.figure(figsize=(5,5))
plt.scatter(Y_test, linear_Y_test_pred, c='b')
z = np.polyfit(Y_test, linear_Y_test_pred, 1)
p = np.poly1d(z)
plt.plot(Y_test, p(Y_test), 'r-')
plt.xlabel('Predicted molecule solubility')
plt.ylabel('Measured molecule solubility')
plt.title("Predicted molecule log(solubility) for training compounds with measured molecule log(solubility)")
plt.show()

#Linear Regression Equation 
print('LogS = %.2f %.2f clogP %.4f MWT + %.4f RB %.2f HBD + %.2f HBA  %.2f TPSA  %.2f AP + %.2f NCP' % (linear.intercept_, linear.coef_[0], linear.coef_[1], linear.coef_[2], linear.coef_[3], linear.coef_[4], linear.coef_[5], linear.coef_[6], linear.coef_[7]))

#Creating Neural network model - model type is artificial neural network ANN rather than GNN
from keras.models import Sequential
from keras.layers import Dense
ann = Sequential()
ann.add(Dense(60, input_dim=8, activation='tanh'))
ann.add(Dense(40, input_dim=60, activation='tanh'))
ann.add(Dense(20, input_dim=40, activation='tanh'))
ann.add(Dense(7, input_dim=20, activation='tanh'))
ann.add(Dense(1, input_dim=7, activation='linear'))

tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
ann.compile(loss='mean_squared_error', optimizer='adam', metrics='mse')
ann.summary

#Fitting/ tuning the model 
ann.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.10, verbose=True)

#Applying the model to the training data set 
ann_Y_train_pred = ann.predict(X_train)
ann_Y_train_pred

#Calculating RMSE
rmse_ann_Y_train_pred = sqrt(mean_squared_error(Y_train, ann_Y_train_pred))
rmse_ann_Y_train_pred
print('Training rmse: ' + str(rmse_ann_Y_train_pred))

#Calculating R^2 Score 
ann_Y_train_score = r2_score(Y_train, ann_Y_train_pred)
ann_Y_train_score
print('Training ANN r^2:'+ str(ann_Y_train_score))

#Test data set 
ann_Y_test_pred = ann.predict(X_test)
ann_Y_test_pred

#Calculating RMSE
rmse_ann_Y_test_pred = sqrt(mean_squared_error(Y_test, ann_Y_test_pred))
rmse_ann_Y_test_pred
print('Training rmse: ' + str(rmse_ann_Y_test_pred))

#Calculating R^2 Score
ann_Y_test_score = r2_score(Y_test, ann_Y_test_pred)
ann_Y_test_score
print('Testing ANN r^2:'+ str(ann_Y_train_score))

#Molecule number array for ANN
mol_numbers = np.arange(1, 230, 1).reshape(229,1)
mol_numbers

#Plotting Test neural network against test linear regression model to compare accuracy of results 
plt.figure(figsize=(10,10))
plt.scatter(mol_numbers, Y_test, c='b', marker='s', label="measured_value")
plt.scatter(mol_numbers, ann_Y_test_pred, c='r', marker='^', label="ANN value")
plt.scatter(mol_numbers, linear_Y_test_pred, c='g', marker='o', label="Linear Regression value")
plt.xlabel("Molecule")
plt.ylabel("Measured log(solubility:mol/L)")
plt.title("Distribution of measured molecule solubility and predicted molecule solubility comparing Linear Regression and ANN")
plt.legend()
plt.show()