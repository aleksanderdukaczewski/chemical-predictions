import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('./datasets/processed_esol.csv', delimiter=",")

a = []

X = df[[
      "LogP", 
      "AromaticProportion",
      "Molecular Weight",
      "Number of Rotatable Bonds", 
      "Number of H-Bond Donors", 
      "Number of Rings", 
      "Polar Surface Area",
      "TotalHalogenAtoms"
]]
Y = df["measured log solubility in mols per litre"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

print("Train:")
Y_pred_train = model.predict(X_train)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))

print("\nTest:")

Y_pred_test = model.predict(X_test)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))

yintercept = '%.2f' % model.intercept_
LogP = '%.2f LogP' % model.coef_[0]
AP = '%.2f AP' % model.coef_[1]
MW = '%.4f MW' % model.coef_[2]
RB = '%.4f RB' % model.coef_[3]
HB = '%d HB' % model.coef_[4]
RINGS = '%d RINGS' % model.coef_[5]
SA = '%.3f SA' % model.coef_[6]
HA = '%d HA' % model.coef_[7]


print('LogS = ' + 
      ' ' + 
      yintercept + 
      ' ' + 
      LogP + 
      ' ' + 
      MW + 
      ' ' + 
      RB + 
      ' ' + 
      AP + 
      ' ' + 
      HB + 
      ' ' +
      RINGS +
      ' ' + 
      SA +
      ' ' +
      HA
)

full = linear_model.LinearRegression()
full.fit(X, Y)

print("\nFull dataset:")

full_pred = model.predict(X)
print('Coefficients:', full.coef_)
print('Intercept:', full.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y, full_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y, full_pred))

full_yintercept = '%.2f' % full.intercept_
full_LogP = '%.2f LogP' % full.coef_[0]
full_MW = '%.4f MW' % full.coef_[1]
full_RB = '+ %.4f RB' % full.coef_[2]
full_AP = '%.2f AP' % full.coef_[3]
full_HB = '%d HB' % model.coef_[4]
full_RINGS = '%d RINGS' % model.coef_[5]
full_SA = '%.3f SA' % model.coef_[6]
full_HA = '%d HA' % model.coef_[7]
print('LogS = ' + 
      ' ' + 
      full_yintercept + 
      ' ' + 
      full_LogP + 
      ' ' + 
      full_MW + 
      ' ' + 
      full_RB + 
      ' ' + 
      full_AP + 
      ' ' + 
      full_HB + 
      ' ' + 
      full_RINGS + 
      ' ' + 
      full_SA +
      ' ' +
      full_HA)

# print(Y_train.shape, Y_pred_train.shape)
# print(Y_test.shape, Y_pred_test.shape)


# Plot the results

plt.figure(figsize=(11,5))

# 1 row, 2 column, plot 1
plt.subplot(1, 2, 1)
plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.3)

z = np.polyfit(Y_train, Y_pred_train, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

# 1 row, 2 column, plot 2
plt.subplot(1, 2, 2)
plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(Y_test, Y_pred_test, 1)
p = np.poly1d(z)
plt.plot(Y_test,p(Y_test),"#F8766D")

plt.xlabel('Experimental LogS')

plt.savefig('plot_horizontal_logS.png')
plt.savefig('plot_horizontal_logS.pdf')
plt.show()