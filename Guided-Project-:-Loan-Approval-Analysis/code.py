# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
path = pd.read_csv(path)
bank = pd.DataFrame(path)
print(bank.head())
categorical_var = bank.select_dtypes(include='object')
print(categorical_var.head())

numerical_var = bank.select_dtypes(include='number')
print(numerical_var)




# code ends here


# --------------
# code starts here
banks = bank.drop('Loan_ID',axis=1)
print(banks.isnull().sum())
bank_mode = banks.mode()
banks = banks.fillna(bank_mode.iloc[0])
print(banks.head())
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks,index=['Gender','Married','Self_Employed'],
values='LoanAmount',aggfunc='mean')
print(avg_loan_amount)


# code ends here



# --------------
# code starts here
# print(banks.head())
loan_approved_se = banks[(banks['Self_Employed']=="Yes") & (banks['Loan_Status']
=="Y")]
print(loan_approved_se)
loan_approved_nse = banks[(banks['Self_Employed']=="No") & (banks['Loan_Status']
=="Y")]
print(loan_approved_nse)
percentage_se = (len(loan_approved_se)/614) *100
print(percentage_se)
percentage_nse = (len(loan_approved_nse)/614) *100
print(percentage_nse)
# code ends here


# --------------
# code starts here
loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12)
# print(len(loan_term))
big_loan_term = len(loan_term[loan_term>=25])
print(big_loan_term)


# code ends here


# --------------
# code starts here
column = ['ApplicantIncome','Credit_History']
loan_groupby = banks.groupby(['Loan_Status'])[column]

mean_values = loan_groupby.agg([np.mean])
print(mean_values)

# code ends here


