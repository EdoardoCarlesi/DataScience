import pandas as pd
import numpy as np

f = 'dati/ecommerce'

df = pd.read_csv(f)

print(df.head())
print(df.info())
#print(df['Purchase Price'].mean())
#print(df['Purchase Price'].min())
#print(df['Purchase Price'].max())
#print(len(df[df['Language'] == 'en']))
#print(len(df[df['Job'] == 'Lawyer']))
#print(len(df[df['AM or PM'] == 'AM']))
#print(len(df[df['AM or PM'] == 'PM']))
#print(df['Job'].value_counts().head(5))
#print(df[df['Lot'] == '90 WT']['Purchase Price'])

#print(df[df['Credit Card'] == 4926535242672853]['Email'])

def parse_year(date):
    return date[3:]

def parse_mail(mail):
    mSplit = mail.split("@", 1)
    provider = mSplit[1]
    #print(provider)
    return provider

#print(df[df['CC Exp Date'].apply(lambda x: parse_year(x)) == '25'].apply(len))
#print(df[df['Email'].apply(lambda x: parse_mail(x))].value_counts().head(5))
df['Providers'] = df['Email'].apply(lambda x: parse_mail(x))
print(df['Providers'].value_counts().head(5))

'''
selectedIds = df['BasePay']#.astype(str)#.str.isdigit()#.str.isalnuma() #fillna(False)
#print(selectedIds.str.isdigit())

#selectedIds = df['BasePay'].notnull()
#allPay = df['BasePay'][selectedIds]

print(len(selectedIds))

notProv = 'Not Provided'

    VERY IMPORTANT: HOW TO CONVERT TO NUMERIC VALUES NaNs BY FORCE !!!!!!!!!!!!
checkStr = 'TotalPayBenefits'

#df['OvertimePay'] = pd.to_numeric(df['OvertimePay'], errors='coerce').fillna(0.0)
df[checkStr] = pd.to_numeric(df[checkStr], errors='coerce').fillna(0.0)
print(df[checkStr].min())
maxPay = df[checkStr].min()

print(df['EmployeeName'][df[checkStr].eq(maxPay)])

checkStr = 'JobTitle'
nJobs = len(df[checkStr].unique())
#allJobs = df[checkStr].unique().tostring()
#allJobs = df[checkStr].value_counts().head(5)
#allJobs = sum(df[df['Year']== 2013][checkStr].value_counts() == 1)
allJobs = sum(df[checkStr].apply(lambda x: find_string(x)))
print((allJobs))

df['titleLen'] = df[checkStr].apply(len)

print(df['titleLen'])
#c = df[['titleLen', 'BasePay']].corr()
#c = 
#print(df[['titleLen', 'TotalPayBenefits']].corr())
print(df[['titleLen', 'OvertimePay']].corr())
#print(c)

#n = df[checkStr].str.count(allJobs)
#n = df[checkStr][df[checkStr].str.count(allJobs)]
#print(n)





#print(df['TotalPayBenefits'][df['EmployeeName'].str.match('JOSEPH DRISCOLL')])
#print(remStr['OvertimePay'].astype(str).str.isdigit())

#allPay = df['BasePay']
#print(allPay.mean())
#allPay = df[df['OvertimePay'].str.isdigit()]
#print(df['OvertimePay'].str.isdigit().fillna(False))
#ids = df['OvertimePay'].str.isdigit().fillna(False)
#print(df['OvertimePay'])
#print(df['OvertimePay'][ids])

#for pay in allPay:
#    print('Pay:', pay)   


#pay = df['OvertimePay'].max()
#print(pay)
#print(pay)
#truePay = pay[pay.notnull()]
#truePay = pay[pay.isna()]
'''

'''
print('Pay: ', type(pay))
print('tPay: ', type(truePay))
print('tPay: ', len(truePay))
print('tPay: ', np.mean(truePay))
'''


#print(truePay.)
#nTrue = truePay.size()
#print(truePay, nTrue)
