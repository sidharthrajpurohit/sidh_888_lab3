# In[1]:


#importing the numpy library
import numpy as np


# In[2]:


#creating a 2d array
array_2d = np.array([[], []])

print(array_2d)


# In[3]:


# creating matrix class with the attribute 2d array
class matrix():
    attri_1=array_2d
    


# In[4]:


def matrix():
    file = open("Data.csv")
    array_2d = np.loadtxt(file, delimiter=",")
    return array_2d


# In[5]:


#save the dataset into the 2d array
file = open("Data.csv")
array_2d = np.loadtxt(file, delimiter=",")


# In[6]:


print(array_2d)


# In[7]:


#save the 2d array into m variable
m=array_2d


# In[8]:


m


# # Standardization of 2d array

# In[9]:


m.ndim


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler=StandardScaler()


# In[12]:


scaler.fit_transform(m)


# In[13]:


m_scaled=scaler.fit_transform(m)


# In[14]:


col=m_scaled[:,0]


# In[15]:


np.var(col)


# # get_distance function

# In[16]:


A = np.array([[], []])
def get_distance(beta,m,A):
    A= np.linalg.norm(beta - m[1,:],axis=0)
    print(A)
print(get_distance(3,m,A))
     
        


  


# In[17]:


print(get_distance(6,m,A))


# # get_count_frequency

# In[18]:


d = dict(enumerate(m.flatten(), 1))


# In[19]:


d


# In[20]:


def get_count_frequency():
    elements_count = {}
    for element in d:
        if element in elements_count:
            elements_count[element] += 1
        else:
            elements_count[element] = 1
    for key, value in elements_count.items():
        print(f"{key}: {value}")


# In[21]:


print(get_count_frequency())


# In[22]:


type(m)


# In[23]:


type(d)


# # get_initial_weights

# In[24]:


p = np.random.rand(10)
p


# In[25]:


def get_initial_weights(p):
    p = np.random.rand(10)
    print(p)

    


# In[26]:


print(get_initial_weights(p))


# # get_centroids function

# In[27]:


m


# In[28]:


S = np.array([[], []])


# In[29]:


def closest_centroids(m, centroids):  
    x = m.shape[0]
    k = centroids.shape[0]
    S = np.zeros(x)
    for i in range(x):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((m[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                S[i] = j
    return S
        
        


# In[30]:


S


# In[31]:


m.shape


# In[32]:


def compute_centroids(m, S, k):  
    c, n = m.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        indices = np.where(S == i)
        centroids[i,:] = (np.sum(m[indices,:], axis=1) / len(indices[0])).ravel()
    return centroids


# In[33]:


def get_centroids(m, k):  
    c, n = m.shape
    centroids = np.zeros((k, n))
    S= np.random.randint(0, c, k)
    for i in range(k):
        centroids[i,:] = m[S[i],:]
    return centroids
get_centroids(m, 3)


# # get_groups function 

# In[34]:


def get_groups(m,k,beta):
    S=print(get_centroids(m,3))
    d=print(get_distance(3,m,A))
print(get_groups(m,3,2))
    


# # get_new_weights function

# In[35]:


def get_new_weights(m, k):  
    c, n = m.shape
    centroids = np.zeros((k, n))
    S= np.random.randint(0, c, k)
    for i in range(k):
        centroids[i,:] = m[S[i],:]
    return centroids
get_new_weights(m, 3)
   

 

    


# In[36]:


def run_test():
    for k in range(2,5):
        for beta in range(11,25):
            S = get_groups(m, k, beta/10)

print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))


# In[37]:


print(run_test())


# In[ ]:





# In[ ]:




