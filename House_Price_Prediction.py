#!/usr/bin/env python
# coding: utf-8

# ### About the data:
# 
# The real estate markets, like those in Sydney and Melbourne, present an interesting opportunity for data analysts to analyze and predict where property prices are moving towards. Prediction of property prices is becoming increasingly important and beneficial. Property prices are a good indicator of both the overall market condition and the economic health of a country.
# 
# #### Columns:
# 
# - Date: Date when the house is ready for sale. 
# - Price: Price of the house to be sold.
# - Bedrooms: No. of bedrooms in the house.
# - Bathrooms: No. of bathrooms in the house.
# - Sqft_living: Squarefoot of Living in the house.
# - Sqft_lot: Squarefoot of Floor in the house. 
# - Floors: Floors on which living area located. 
# - Waterfront: If waterfront available in front of house.
# - View: Vie from the house.
# - Condition: Condition of the house.
# - Sqft_above: Squarefoot above is the space available at roof. 
# - Sqft_basement: Squarefoot basement is the space available at the basement.
# - Yr_built: In which year the house is built.
# - Yr_renovated: Year of renovation.
# - Street: On which street house is located.
# - City: City in which the country is located.
# - Statezip: Zip code of the area in which house is located.
# - Country: Country is US.

# **Task**: `Here the task is to predict the price of house(dependent variable) located in the cities of US, with the help of other essential features (independent variable) available in our dataset.`

# ### Regression
# 
# `Regression analysis consists of a set of machine learning methods that allow us to predict a continuous outcome variable (Y) based on the value of one or multiple predictor variables (X).`

# In[1]:


#importing basic libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **%matplotlib inline** : This is known as magic inline function.
# When using the 'inline' backend, our matplotlib graphs will be included in our notebook, next to the code. 

# Let's dive into our dataset.

# In[2]:


data = pd.read_csv("data.csv")


# In[3]:


df = data.copy()
df.head()


# `Now firstly let's get a description about our dataset.`

# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df.describe()


# Checking our model accuracy in the starting.

# In[8]:


X = df.drop(["date",'street', 'city','statezip', 'country','price'], axis=1)
y = df[['price']]
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X,y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
mlrm = LinearRegression()

mlrm.fit(train_X,train_y)

mlrm.score(train_X, train_y)*100


# Poor score, but what do we expect from raw data.

# Let's start **Exploratory Data Analysis(EDA)**.

# In[9]:


df.corr()


# So, our price is majorly dependent upon bedrooms, bathrooms, sqft_living among these continuous variables.

# In[10]:


df.isnull().sum()


# In[11]:


df[df==0].count()


# I've checked for NaN values first and then after go to check for the 0s in our dataset.
# 
# Many of you might think why should I checked for values equals to 0, because sometimes our data is fill with unconditional 0 values inspite of NaN values. So, it must be our habit to check for 0s also while cleaning our dataset.

# Now, some of the columns like waterfront, view, sqft_above and sqft_basement seems more authentic to be 0s in their values because a house might not have waterfront or any view or basement area and might not be ever renovated.
# 
# But, it is really hard to digest a house with 0 bedrooms and 0 bathrooms, and impossible to have any house with 0 price, who'll sell a house for free.😂🤷‍♀️
# 
# Let's look into it with closer insights.

# In[12]:


df[df["price"]==0].head(50)


# In[13]:


plt.figure(figsize=(15,6))
ax = sns.distplot(df[df["price"]==0].sqft_living)
ax.set_title('Sqft_living for 0 price', fontsize=14)


# Majority of the 0 price house's sqft_living ranges between 1000-5000.
# 
# We have to check its correlation with other columns also.

# In[14]:


df[df["price"]==0].agg([min, max, 'mean', 'median'])


# So, 0 price houses have ~4 bedrooms, ~2.5 bathrooms and ~2800 sqft living.
# 
# As we discussed above these are some major columns on which price depends. So we'll use these to replace 0 prices of houses.

# In[15]:


df1 = df[(df.bedrooms == 4) & (df.bathrooms > 1) & (df.bathrooms < 4) & (df.sqft_living > 2500) & 
         (df.sqft_living < 3000) & (df.floors < 3) & (df.yr_built < 1970)]


# In[16]:


df1.shape


# In[17]:


df1.price.mean()


# In[18]:


df['price'].replace(to_replace = 0, value = 735000, inplace = True)
len(df[(df['price'] == 0)])


# And, it's done!!

# Before moving forward let's quickly deal with 0 bedrooms and bathrooms.

# In[19]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['bedrooms'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bedrooms VS Price', fontsize=14)


# In[20]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['bedrooms'], y=df['sqft_living'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bedrooms VS Sqft_living', fontsize=14)


# As you can also observe sqft living and prices of 8 bedrooms closely related to each other so, let's replace 0 bedrooms with 8

# In[21]:


df['bedrooms'].replace(to_replace = 0, value = 8, inplace = True)
len(df[(df['bedrooms'] == 0)])


# In[22]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['bathrooms'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bathrooms VS Price', fontsize=14)


# In[23]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['bathrooms'], y=df['sqft_living'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bathrooms VS Sqft_living', fontsize=14)


# As you can also observe sqft living and prices of 3.25 bathrooms closely related to each other so, let's replace 0 bathrooms with 3.25

# In[24]:


df['bathrooms'].replace(to_replace = 0, value = 3.25, inplace = True)
len(df[(df['bathrooms'] == 0)])


# Alright then, we've dealt with 0s let's move forward

# In[25]:


ax = sns.pairplot(df)


# There are alot of outliers can clearly observed!!
# 
# Let's visualize it more clearly!

# In[26]:


plt.figure(figsize=(15,10))
ax = sns.histplot(df['price'], kde=True)
ax.set_title('Histplot of Price', fontsize=14)


# Highly negative skewed!! Can be treated by eleminating or replacing outliers.
# 
# **Outliers**: Outliers are extreme values that deviate from other observations on data , they may indicate a variability in a measurement, experimental errors or a novelty. In other words, an outlier is an observation that diverges from an overall pattern on a sample.
# 
# **How to detect them**:
# - Scatter plot
# - Box plot
# - Z score
# 
# **Treatment**:
# - Z Score
# - IQR
# - Eleminating
# - Central Tendency Capping.
# 
# **Z Score**: 
# The z-score or standard score of an observation is a metric that indicates how many standard deviations a data point is from the sample’s mean, assuming a gaussian distribution. This makes z-score a parametric method. Very frequently data points are not to described by a gaussian distribution, this problem can be solved by applying transformations to data ie: scaling it.

# In[27]:


from scipy import stats
df['price'] = df['price'].replace([data['price'][np.abs(stats.zscore(data['price'])) > 3]],np.median(df['price']))


# Handling outliers of price columns. As, it is very important because at last we're predicting price and disturbance in its distribution might hit hard to our predictions.

# In[28]:


plt.figure(figsize=(15,10))
ax = sns.histplot(df['price'], kde=True)
ax.set_title('Histplot of Price', fontsize=14)


# What a CHANGE!!
# Quickly Visualizing more columns.

# In[29]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=df, x="sqft_living", y="price")
ax.set_title('Sqft_living VS Price', fontsize=14)


# As we can observe that many of the data is less than 6000. So, let's handle this uneven data.
# 
# Removing outliers of Sqft_living using different approach.

# In[30]:


df['sqft_living'] = np.where((df.sqft_living >6000 ), 6000, df.sqft_living)


# In[31]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=df, x="sqft_living", y="price")
ax.set_title('Sqft_living VS Price', fontsize=14)


# Similarly, doing with rest of the continuous columns.

# In[32]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=df, x="sqft_lot", y="price")
ax.ticklabel_format(style='plain')
ax.set_title('Sqft_lot VS Price', fontsize=14)


# Too much disturbance in this column and also as we checked above is not much correlated to our dependent column price. So, I've decided to leave it as it is.

# In[33]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=df, x="sqft_above", y="price")
ax.ticklabel_format(style='plain')
ax.set_title('Sqft_above VS Price', fontsize=14)


# In[34]:


df['sqft_above'] = np.where((df.sqft_above >5000 ), 5000, df.sqft_above)


# In[35]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=df, x="sqft_basement", y="price")
ax.ticklabel_format(style='plain')
ax.set_title('Sqft_basement VS Price', fontsize=14)


# Alot of 0's in sqft_basement but not an outliers because houses might have without basements.
# 
# **Note**: `not all extreme points can be treated as outlier. May they are really an authentic data.`
# 
# Still, let's make extreme large values near 2000.

# In[36]:


df['sqft_basement'] = np.where((df.sqft_basement >2000 ), 2000, df.sqft_basement)


# Let's catch up with discrete variables.

# In[37]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['bedrooms'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bedrooms VS Price', fontsize=14)


# In[38]:


df['bedrooms'].nunique()


# In[39]:


bedroom = df.groupby(['bedrooms']).price.agg([len, min, max])
bedroom


# Max price of 7,8,9 bedrooms are less than that of 6 bedrooms which is kind strange, but it might possible if the house is far from the main town or have some issues like paranormal activities like hollywood movies😂😂!!
# 
# We can leave it as it ease or we can change it also!! I personally think let's deal with it because less than 20 rows have more than 6 bedrooms and it will might lead to disturbance in our distribution.

# In[40]:


df['bedrooms'] = np.where((df.bedrooms >6 ), 6, df.bedrooms)


# In[41]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['bathrooms'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Bathrooms VS Price', fontsize=14)


# In[42]:


df['bathrooms'].nunique()


# In[43]:


bathroom = df.groupby(['bathrooms']).price.agg([len , min, max])
bathroom


# See, personally I don't have idea about 1.25, 1.75 kind of bethrooms but it doesn't seems an error because we have alot more entries like that.

# In[44]:


df['bathrooms'] = np.where((df.bathrooms == 0.75), 1, df.bathrooms)
df['bathrooms'] = np.where((df.bathrooms == 1.25 ), 1, df.bathrooms)
df['bathrooms'] = np.where((df.bathrooms > 4.75 ), 5, df.bathrooms)


# I here replaced few of less occuring entries to decrease the skewness of our distribution.

# In[45]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['floors'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Floors VS Price', fontsize=14)


# Maximum price is for 2.5 floors trailing by 3.5 floors.

# In[46]:


floor = df.groupby(['floors']).price.agg([len , min, max])
floor


# Only two houses have 3.5 floors. Handling this by replacing it with 3.0 floors.

# In[47]:


df['floors'] = np.where((df.floors == 3.5 ), 3, df.floors)


# In[48]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['waterfront'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Waterfront VS Price', fontsize=14)


# In[49]:


waterfront = df.groupby(['waterfront']).price.agg([len , min, max])
waterfront


# Clearly, houses with waterfront starts only with ~0.4million. No, changes for me in this column.

# In[50]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['view'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('View VS Price', fontsize=14)


# In[51]:


view = df.groupby(['view']).price.agg([len , min, max])
view


# Well no need for any changes. 0 view is for houses having no sea facing view or anything like that. So it seems pretty authentic to me.

# In[52]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=df['condition'], y=df['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Condition VS Price', fontsize=14)


# In[53]:


condition = df.groupby(['condition']).price.agg([len , min, max])
condition


# Possibly the condition here is for the condition of house and I assuming 5 as best condition and 1 as the worst.
# 
# But 1 has just 6 houses that means USA is a country of good conditioned houses.
# 
# Well just one change for me to replace 1 with 2. That's all.

# In[54]:


df['condition'] = np.where((df.condition == 1 ), 2, df.condition)


# Done with the Discrete columns here. Yr_built & yr_renovated are out of two least correlated columns with price. Thus, decided to leave them as it is.

# Let's quickly check prediction with these continuous columns only except sqft_lot, yr_built and yr_renovated.

# In[55]:


X = df.drop(["date",'street', 'city','statezip','sqft_lot','country','price','yr_built','yr_renovated'], axis=1)
y = df[['price']]
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X,y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
mlrm = LinearRegression()

mlrm.fit(train_X,train_y)

mlrm.score(train_X, train_y)*100


# Increased!!...

# Time to drop few columns from our dataframe.

# In[56]:


df.drop(["date",'yr_built','yr_renovated','sqft_lot'], axis=1, inplace = True)


# Checking for multicollinearity among the continuous columns using VIF methods.
# 
# **Multicollinearity**: Multicollinearity occurs when two or more independent variables are highly correlated with one another in a regression model.
# 
# **Why not Multicollinearity?**: Multicollinearity can be a problem in a regression model because we would not be able to distinguish between the individual effects of the independent variables on the dependent variable.
# 
# **Detection of Multicollinearity**: Multicollinearity can be detected via various methods. One of the popular method is using VIF.
# 
# **VIF**: VIF stands for Variable Inflation Factors. VIF determines the strength of the correlation between the independent variables. It is predicted by taking a variable and regressing it against every other variable.
# 
# `Here, I'll check VIF only for the continuous variables.`

# In[57]:


X1 = df.drop(['street', 'city','statezip','country'], axis=1)


# In[58]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = add_constant(X1)

pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)


# Now, VIF of sqft_living & sqft_above is very very high. That means we have to drop any two of the columns because it's not at all good for our model. 
# 
# But Wait! How will we decide which of the columns should be dropped?
# 
# Here comes the role of Significancy.
# 
# **Significancy**: In statistics, statistical significance means that the result that was produced has a reason behind it, it was not produced randomly, or by chance.
# 
# Here, we are currently focusing on continuous data. So best statistical test according to this condition will be: Correlation Coefficients
# 
# **Correlation Coefficients**: Correlation coefficients are used to measure how strong a relationship is between two variables.

# In[59]:


df.corr()


# In[60]:


plt.figure(figsize=(15,6))
ax = sns.heatmap(df.corr(),annot = True)
ax.set_title('CORRELATION MATRIX', fontsize=14)


# Sqft_living and sqft_above are highly correlated but Sqft_abobe is less correlated with price. So we got our column which can be dropped.

# In[61]:


X_vif = X_vif.drop(['sqft_above'],axis = 1)
pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)


# Perfect👌!!

# Now, as you all can notice in heatmap. Columns like condition and waterfront have less impressive correlation with our dependent variable. So that I've decided to drop them as well.

# In[62]:


df.drop(['waterfront','condition','sqft_above'],axis=1, inplace=True)


# In[63]:


df.head()


# In[64]:


df.dtypes


# Here is the end of EDA with continuous terms.

# Left with object types to handle. Let's start with country.

# In[65]:


df['country'].nunique()


# All the entries have common country, will never affect our dependent variable. Will drop it later.

# In[66]:


df['street'].nunique()


# Too many unique values for 4600 entries, again will never affect our dependent variable. Will drop it later.

#  

# Just left with city and statezip. Here we all know satezip and city are quite relateable things. Like every city has a unique zip code. So, we have to choose wisely one out of these two columns.

# In[67]:


df['city'].nunique()


# In[68]:


df['statezip'].nunique()


# As we can observe, city has 44 unique values and statezip has 77. Let's see which variable has more impact to price.

# In[69]:


city = df.groupby(['city']).price.agg([len, min, max])
pd.set_option('display.max_rows',70)
city


# In[70]:


plt.figure(figsize=(15,10))
ax = sns.barplot(x="city", y="price", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right");


# In[71]:


statezip = df.groupby(['statezip']).price.agg([len, min, max])
pd.set_option('display.max_rows',70)
statezip


# In[72]:


plt.figure(figsize=(15,10))
ax = sns.barplot(x="statezip", y="price", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right");


# You can observe a litle high variation in statezip in comparison to city. So I'have selected our final column that is statezip.
# 
# Now, let's drop street, city and country before moving forward.

# In[73]:


df.drop(['street','city','country'],axis=1, inplace=True)


# **Encoding**: In laymens language it is just converting data into numerical forms so that our model can understand data easily.
# 
# There are many techniques for label encoding but here after observing data well I choose One Hot Encoding.
# 
# **Note**: The one hot encoder does not accept 1-dimensional array or a pandas series, the input should always be 2-dimensional and the data passed to the encoder should not contain strings.

# In[74]:


df = pd.get_dummies(df, columns=['statezip'], prefix = ['statezip'])

df.head()


# In[75]:


df.shape


# In[76]:


df.columns


# In[77]:


X1 = df.drop(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view',
       'sqft_basement'],axis = 1)
y = df["price"]


# In[78]:


import scipy.stats as stats
for i in X1.columns:
    print(stats.f_oneway(X1[i],y))


# None of the pvalue is greater than significance value 0.05. So, it proves there is no relationship between the variables with y, so must not be dropped.

# Now, we will train our model divide for train, test and validation dataset.
# 
# **Training Dataset**: The sample of data used to fit the model.
# 
# **Validation Dataset**: The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.
# 
# **Test Dataset**: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
# 
# Apply Linear Regression and hence, it's done.

# ![image.png](attachment:image.png)

# In[79]:


X = df.drop(["price"],axis = 1)
y = df["price"]

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y= train_test_split(X,y, test_size=0.2, random_state=8)

from sklearn.linear_model import LinearRegression
mlrm = LinearRegression()

mlrm.fit(train_X,train_y)

mlrm.score(train_X, train_y)*100


# In[80]:


X_val, X_test, y_val, y_test = train_test_split(test_X, test_y, test_size=0.1, random_state=42)


# In[81]:


mlrm.fit(train_X,train_y)


# In[83]:


mlrm.score(X_test, y_test)


# In[87]:


import statsmodels.api as sm
mod = sm.OLS(train_y, train_X)
res = mod.fit()
print(res.summary())


# In[85]:


plt.figure(figsize=(16,8))
#plt.subplot(211)
plt.plot(test_y.reset_index(drop=True), label='Actual', color='g')
#plt.subplot(212)
plt.plot(mlrm.predict(test_X), label='Predict', color='r')
plt.legend(loc='upper right')


# We can see the predicted line have almost covered the green line very well. Great!!
# 
# Also, score seems quite good!! Almost 80% values are predicted well. So, Congrats to us!!😊👍
