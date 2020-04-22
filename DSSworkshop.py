#!/usr/bin/env python
# coding: utf-8

# # Election Workshop

# In[1]:


#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16,8)
plt.rcParams['figure.dpi'] = 150
sns.set()

from IPython.display import display, Latex, Markdown


# ## Primary Prediction
# Predict the 2020 United States Democratic presidential primary election results.

# ### Getting 2020 Democratic Primary poll data
# We can use the pandas read_html function to scrap the tables from https://www.realclearpolitics.com/epolls/latest_polls/democratic_nomination_polls/. 

# In[2]:


polls20 = pd.read_html('https://www.realclearpolitics.com/epolls/latest_polls/democratic_nomination_polls/')

for t in polls20:
    display(t)


# Here, we're going to quickly combine all the tables in this list `polls20` that are not empty; then filter out the rows that are not state-wide polls.

# In[3]:


poll20 = pd.DataFrame(columns=[0,1,2,3])

for t in polls20:
    if (len(t) != 1): 
        poll20 = poll20.append(t.iloc[1:])

poll20 = (poll20.rename(columns={0:'Race', 1:'Poll', 2:'Results', 3:'Spread'})
            .query("Race != '2020 Democratic Presidential Nomination'")
            .reset_index().iloc[:, 1:])
display(poll20)


# ### Creating new columns

# How do we extract the name of the leading candidate in each poll? 

# In[4]:


poll20['Race'].str.extract(r'([\w ]+) Democratic')


# In[5]:


poll20['State'] = poll20['Race'].str.extract(r'([\w ]+) Democratic')
poll20.head()


# In[6]:


poll20['Spread'].str.extract(r'([A-z]+)')


# In[7]:


poll20['Poll_adv'] = poll20['Spread'].str.extract(r'([A-z]+)')
poll20.head()


# Here we can crudely calculate the candidate that's leading in the state polls. 

# In[8]:


poll20['Poll_adv'].value_counts() / len(poll20)


# Hmm, Sanders has a huge advantage here. However, it doesn't seem to hold for the Super Tuesday result... what may have gone wrong? 

# ### Biden vs. Sanders: Getting the Accurate Numbers

# Super Tuesday: a day on which several US states hold primary elections.
# #### Getting Super Tuesday Data
# Suprising turns happened on Super Tuesday -- Biden won many states that are supposed to be taken over by Sanders. What happens if we combine the Primary voting data up to Super Tuesday with our polling data? Who will win? 

# We are using the data from https://www.realclearpolitics.com/epolls/2020/president/democratic_delegate_count.html. Because the table on this page is loaded with JavaScript and cannot be easily scraped, we have downloaded a static html page beforehand inside the elections-sp20 folder. 
# 
# Here we will load the table from the html file. 

# In[9]:


st20_original = pd.read_html('super_tuesday.html')[0]
st20_original


# A few lines of code to perform a few data cleaning tasks. 

# In[10]:


# Process the original table for delegates... we will use this series later! 
st20_original = st20_original.drop([0, 58])
st20_original['Delegates'] = st20_original['Delegates*'].str.extract(r'(\d+)\s\(').astype('float')
st20_original = st20_original.drop(columns=['Delegates*'])
# Replace the State column with correct names
st20_original['State'] = st20_original['State'].str[2:]

# Only select States that already had votes and drop empty columns
st20 = st20_original.loc[~st20_original['Biden'].isna(), ~st20_original.loc[1].isna()].set_index('State').fillna(0)
st20


# Now we're trying to get a more accurate prediction of the Democratic primary. We can combine the poll data and the Super Tuesday data, and try to predict the # of delegates for Biden and Sanders in each state.

# In[11]:


poll20.head()


# In[12]:


# Separate the percentage of Biden and Sanders from the Results column
poll20['Biden%']=poll20['Results'].str.extract(r'Biden\s(\d+)').astype('float')
poll20['Sanders%']=poll20['Results'].str.extract(r'Sanders\s(\d+)').astype('float')
poll20.head()


# In[13]:


# Create a new table based on the polling data for each state, using average of the percentages
poll20_BvS = poll20[['State', 'Biden%', 'Sanders%']].groupby('State').mean()
poll20_BvS


# In[14]:


# Combining the # of delegates in each state with the poll %
delegates = st20_original[['State', 'Delegates']].set_index('State')
BvS = poll20_BvS.join(delegates, how = 'outer')
BvS


# In[15]:


# Calculate the "prediction" delegate counts for Biden and Sanders
BvS['Biden'] = BvS['Delegates'] * BvS['Biden%'] * 0.01
BvS['Sanders'] = BvS['Delegates'] * BvS['Sanders%'] * 0.01
BvS


# In[16]:


# Update the columns with Super Tuesday Data, and drop rows with no rows
BvS['Biden'].update(st20['Biden'])
BvS['Sanders'].update(st20['Sanders'])
BvS = BvS[~BvS['Biden'].isna()]
BvS


# Summing the counts of delegates up, who's having a lead?  

# In[17]:


print('Biden: approx. # of delegates: ', BvS['Biden'].sum(), '%:', BvS['Biden'].sum() / BvS['Delegates'].sum())
print('Sanders: approx. # of delegates: ', BvS['Sanders'].sum(), '%:', BvS['Sanders'].sum() /BvS['Delegates'].sum())


# ## The Electoral College
# 
# The US president is chosen by the Electoral College, not by the
# popular vote, as we saw in 2016. Each state is alotted a certan number of 
# electoral college votes; this is supposed to represent their population size.
# Whomever wins in the state gets all of the electoral college votes for that state.
# 
# There are 538 electoral college votes-- meaning, in order to be President of the United States,
# one has to win 270 of them.
# 
# 
# 2016 has shown us that we can't always rely on the polls to predict election outcomes;
# the Electoral College is why. 
# 
# In 2016, pollsters correctly predicted the election outcome in 46 of the 50 states. 
# The remaining 4 states accounted for a total of 75 votes, and 
# whichever candidate received the majority of the electoral college votes in these states would win the election. 
# 
# These states were <strong>Florida, Michigan, Pennsylvania,</strong> and <strong>Wisconsin.</strong>
# 
# These states specifically have proven to be particularly important because they are <strong>swing states</strong> (i.e. they do not always follow partisan politics; sometimes these states are won by the Democratic party, and other times, they are won by the Republican party). According to experts in politics, these four states are looking to serve as battlegrounds in the election, and will likely determine the outcome for 2020 as they did in 2016.<sup>1</sup>
# 
# <i>Note: a beautiful data visualization was made by the [New York Times](https://www.nytimes.com/elections/2016/results/president), and this shows a little bit clearer why
# these states were so important.</i>
# 
# Their electoral college votes are as follows:
# 
# |State |Electoral College Votes|
# | --- | --- |
# |Florida | 29 |
# |Michigan | 16 |
# |Pennsylvania | 20 |
# |Wisconsin | 10|
# 
# Let's assume that all other states vote the same as they did in the 2016 presidential election.
# That means, for a candidate to win the election, he has to win either:
# * Florida + one (or more) other states
# * Michigan, Pennsylvania, and Wisconsin
# 
# 
# 
# <br />
# <br />
# 
# <sup>1</sup><small>[Washington Post article](https://www.washingtonpost.com/politics/the-2020-electoral-map-could-be-the-smallest-in-years-heres-why/2019/08/31/61d4bc9a-c9a9-11e9-a1fe-ca46e8d573c0_story.html): "The 2020 electoral map could be the smallest in years."</small>

# ### Sampling Error: Looking Into Why The Polls Are Not Always Accurate.
# 
# In this section of the notebook, we are going to explore why all the polls predicted for Hillary Clinton to win, even if polls were selected with no bias.
# 
# To do this, we are going to simulate unbiased polling from the results of the 2016 election, and find the probability that Trump wins based on these simulated polls.
# 
# The electoral margins were very narrow in Florida, Michigan, Pennsylvania, and Wisconsin in 2016. Narrow electoral margins can make it hard to predict the outcome given the sample sizes that the polls used. 
# 
# 
# |State | Trump |   Clinton | Total Voters |
# | --- | --- |  --- |  --- |
# |Florida | 49.02 | 47.82 | 9,419,886  | 
# |Michigan | 47.50 | 47.27  |  4,799,284|
# |Pennsylvania | 48.18 | 47.46 |  6,165,478|
# |Wisconsin | 47.22 | 46.45  |  2,976,150|
# 
# 
# Below is a function, `draw_state_sample(N, state)`, that returns a sample (with replacement) from N voters of one of the four crucial states. It returns a 3-element list: `[Trump votes, Clinton votes, All other candidate votes]`.

# In[18]:


#RUN THIS CELL
def draw_state_sample(N, state):
    if state == "florida":
        return np.random.multinomial(N, [0.4902, 0.4782, 1 - (0.4902 + 0.4782)])
    
    if state == "michigan": 
        return np.random.multinomial(N, [0.475, 0.4727, 1 - (0.475 + 0.4727)])

    if state == "pennsylvania":
        return np.random.multinomial(N, [0.4818, 0.4746, 1 - (0.4818 + 0.4746)])
  
    if state == "wisconsin":
        return np.random.multinomial(N, [0.4722, 0.4645, 1 - (0.4722 + 0.4645)])

    raise("invalid state")


# Next is the function `trump_advantage`, which takes in a sample of votes returned by `draw_state_sample`, and returns the advantage Trump has (as a difference in the proportion of votes between Trump and Clinton).

# In[19]:


#RUN THIS CELL
def trump_advantage(voter_sample):
    N = sum(voter_sample)
    percentage_trump = voter_sample[0]/N 
    percentage_clinton = voter_sample[1]/N 
    return percentage_trump - percentage_clinton


# Now, we are going to simulate 100,000 SRS (simple random samples) of 1500 voters for Florida, Michigan, Pennsylvania, and Wisconsin. This simulates polling 1500 voters in these states 100,000 times. 

# In[20]:


#RUN THIS CELL
simulations_f = [trump_advantage(draw_state_sample(1500, "florida")) for i in range(100000)]
simulations_m = [trump_advantage(draw_state_sample(1500, "michigan")) for i in range(100000)]
simulations_p = [trump_advantage(draw_state_sample(1500, "pennsylvania")) for i in range(100000)]
simulations_w = [trump_advantage(draw_state_sample(1500, "wisconsin")) for i in range(100000)]


# Below we've made histograms to show the distributions for each of the state simulations:

# In[21]:


#RUN THIS CELL
plt.hist(simulations_f) 
plt.title('Florida')
plt.ylabel('# of Simulations')
plt.xlabel('Sampling Distribution Advantage')


# In[22]:


#RUN THIS CELL
plt.hist(simulations_m) 
plt.title('Michigan')
plt.ylabel('# of Simulations')
plt.xlabel('Sampling Distribution Advantage')


# In[23]:


#RUN THIS CELL
plt.hist(simulations_p) 
plt.title('Pennsylvania')
plt.ylabel('# of Simulations')
plt.xlabel('Sampling Distribution Advantage')


# In[24]:


#RUN THIS CELL
plt.hist(simulations_w) 
plt.title('Wisconsin')
plt.ylabel('# of Simulations')
plt.xlabel('Sampling Distribution Advantage')


# #### Question 1
# What do you notice about these histograms? Any similarities? Differences?

# <i>YOUR ANSWER HERE</i>

# Now, we are going to calculate the number of times Trump wins in these simulations. Below we've defined a function `trump_wins(N)` for N sample voters who voted in the swing states above. It returns 1 if Trump is predicted to win, an 0 if not.
# 
# For Trump to win the Electoral College vote, he must do either of the following:
# * win Florida and one (or more) other states
# * Michigan, Pennsylvania, and Wisconsin

# In[25]:


def trump_wins(N):
    wins_florida = trump_advantage(draw_state_sample(N, "florida")) > 0
    wins_michigan = trump_advantage(draw_state_sample(N, "michigan")) > 0
    wins_pennsylvania = trump_advantage(draw_state_sample(N, "pennsylvania")) > 0
    wins_wisconsin = trump_advantage(draw_state_sample(N, "wisconsin")) > 0
    if wins_michigan and wins_pennsylvania and wins_wisconsin:
        return 1
    if wins_florida and (wins_michigan or wins_pennsylvania or wins_wisconsin):
        return 1
    return 0


# Below we've defined `percent_trump` as the proportion of 100,000 simulations that predict a Trump victory (i.e. we called `trump_wins(1500)` 100,000 times)
# 
# This number represents the percent chance that a given sample will correctly predict Trump's victory <strong>even if the sample was collected with absoutely no bias</strong>. 

# In[26]:


percent_trump = np.mean([trump_wins(1500) for i in range(100000)])
percent_trump


# 
# **Note: Many laypeople, even well educated ones, assume that this number should be 1. After all, how could a non-biased sample be wrong? This is the type of incredibly important intuition we hope to develop in you throughout this class and your future data science coursework.**

# #### Question 2
# This is the percent change that we predict a Trump victory, even from a non-biased sample. Why is this answer not 1?

# <i>YOUR ANSWER HERE </i>

# #### Question 3
# We've run these simulations as if this is a non-biased sample, because we have the actual voting data.
# 
# Polling data ultimately is biased, as much as we try to limit this. Below, list some ways that polling data can actually be biased (i.e. not representative of our population).

# <i>YOUR ANSWER HERE</i>
