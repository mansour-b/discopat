"""
Model Inference
===============

This example demonstrates how to detect structures with a CNN using ``patrick``.
"""
# ### Imports

# In[1]:

from tqdm import tqdm

from patrick.cnn import FasterRCNNModel
from patrick.core import Movie
from patrick.display import plot_frame
from patrick.repositories import repository_factory

# ### Definitions

# In[42]:


movie_name = "blob_i/density"
model_name = "faster_rcnn_241113_131447"

computing_device = "cpu"
data_source = "osf"
framework = "torch"


# ### Load the images

# In[ ]:


movie_repository = repository_factory(data_source, "input_movies")
movie = movie_repository.read(movie_name)

for frame in movie.frames:
    plot_frame(frame)


# ### Load the model

# In[11]:


model_repository = repository_factory(data_source, "models")

raw_model = model_repository.read(model_name)
model = FasterRCNNModel.from_dict(raw_model)
model.set_device(computing_device)


# ### Compute predictions

# In[ ]:


analysed_frames = [model.predict(frame) for frame in tqdm(movie.frames)]
analysed_movie = Movie(name=movie.name, frames=analysed_frames, tracks=[])


# ### Display predictions

# In[ ]:


for frame in analysed_movie.frames:
    plot_frame(frame)
