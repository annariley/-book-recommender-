import torch
import numpy as np
import pandas as pd, csv
from torch import nn, optim
torch.manual_seed(291)
np.random.seed(291)#

NumBooks = 5 #Constant number of books to recommend per iteration
RecBatches = 300 #Constant number of books to add to the recommendation list for RL
BoostBatch = 100 #Constant number of books to add to the rec list when needed
BoostThreshold = RecBatches-BoostBatch
TotalNumBooks = 9809
NumUsers = 53424
hardcoded = [3,15,18,815,96,4,9,127,14,2458,2738,316,60,5,2992,259,617,237]


#Code adapted and inspired by movie recommender model in lecture 6
#Code adapted and inspired by movie recommender model in lecture 6
class BookRecommenderEmbeddingML(nn.Module):
  def __init__(self, n_users, n_books, emb_dim):
    super(BookRecommenderEmbeddingML, self).__init__()
    self.user_embedding = nn.Embedding(n_users, emb_dim)
    self.book_embedding = nn.Embedding(n_books, emb_dim)
    nn.init.xavier_uniform_(self.user_embedding.weight)
    nn.init.xavier_uniform_(self.book_embedding.weight)
    self.dropout = nn.Dropout(0.25)
  
  def forward(self, samples):
    users = self.user_embedding(samples[:,0]) # gets embedding of users
    users = self.dropout(users)
    books = self.book_embedding(samples[:,1])
    books = self.dropout(books)
    dot = (users * books).sum(1)
    return torch.sigmoid(dot) * 5.5

def find_paired_user(ratings, matrix):
  bestDiff = np.inf
  bestUser = 0
  for i in range(NumUsers): # 5 for now but should be ds_full.n_users
    diff = 0
    for rating in ratings:
      diff = diff + abs(matrix[i][rating[0]] - rating[1])#Matrix will be tensor - may need to accomadate if so adjust future uses
    if diff < bestDiff:
      bestDiff = diff
      bestUser = i
  print(bestUser)
  return bestUser

class User ():
  def __init__(self, ratings, matrix, id, emb_dim, model):
    self.id = id
    self.ratings = ratings
    self.pair_id = find_paired_user(ratings, matrix)
    emb_index = torch.LongTensor([self.pair_id])
    user_feature_vector = model.user_embedding(emb_index) # get feature vector for user 0
    row = matrix[self.pair_id].detach().numpy()
    self.to_recommend = []
    for i in range(TotalNumBooks):
      self.to_recommend.append([i, row[i]])
    self.to_recommend = sorted(self.to_recommend, key=lambda x : x[1]) #From https://stackoverflow.com/a/4174956
    self.to_recommend.reverse()
    self.rl_model = RLModel(user_feature_vector, emb_dim)
    self.optimizer = optim.SGD(self.rl_model.parameters(),1,0.3)
    self.curr_rec_list = self.to_recommend[0:RecBatches]
    self.to_recommend = self.to_recommend[RecBatches:]
  def update_rec_list(self, matrix):
    if len(self.curr_rec_list) < BoostThreshold:
      self.curr_rec_list = self.curr_rec_list + (self.to_recommend[0:BoostBatch])
      self.to_recommend = self.to_recommend[BoostBatch:]
    for i in range(len(self.curr_rec_list)):
      self.curr_rec_list[i][1] = getRec(self, self.curr_rec_list[i][0], matrix)
    self.curr_rec_list =  sorted(self.curr_rec_list, key=lambda x : x[1]) #From https://stackoverflow.com/a/4174956
    self.curr_rec_list.reverse()
  def get_books(self, matrix):
    curr_recommendation = self.curr_rec_list[0:NumBooks]
    self.curr_rec_list = self.curr_rec_list[NumBooks:]
    self.update_rec_list(matrix)
    return curr_recommendation

def lossFunction(self, expected, rating): 
  return abs(expected - rating)

def create_matrix(model):
  index = torch.IntTensor([range(53424)])
  userMat = model.user_embedding(index)[0]
  index = torch.IntTensor([range(TotalNumBooks)])
  old = model.book_embedding(index)[0]
  bookMat = torch.zeros(24,TotalNumBooks)
  for i in range(24):
    bookMat[i] = torch.narrow(old,1,i,1).flatten()
  result = userMat@bookMat
  return torch.sigmoid(result)*5.5

def create_book_feature_matrix(model):
    index = torch.IntTensor([range(TotalNumBooks)])
    return model.book_embedding(index)[0]

class RLModel(nn.Module):
  def __init__(self,user_vector, emb_dim):
    super(RLModel, self).__init__()
    self.embedding = nn.Embedding(1,emb_dim)
    self.embedding.weight = torch.nn.Parameter(user_vector) 
  def forward(self,input, matrix):
    book_vec = matrix[int(input)] #created using create_book_feature_matrix
    userNum = torch.LongTensor([0])

    result = (self.embedding(torch.LongTensor([[0]]))[0] * book_vec.view(1,24)).sum(1)
    return torch.sigmoid(result)
def getRec(user, book_id, b_matrix):
  user.rl_model.eval()
  with torch.no_grad():
   pred = user.rl_model(book_id, b_matrix)
  return pred[0].item()

def improve(swipe, book_id, user, b_matrix):
  user.rl_model.train()
  with torch.enable_grad():
    user.optimizer.zero_grad()
    pred = user.rl_model(book_id, b_matrix)
    loss = nn.MSELoss()
    improve = loss(pred, torch.FloatTensor([swipe]))
    improve.backward(retain_graph=True)
    user.optimizer.step()#

def get_book_data(book_id, collection):
    return collection.find_one({"book_id":book_id})["title"], collection.find_one({"book_id":book_id})["authors"], collection.find_one({"book_id":book_id})["image_url"]

def get_recs(users, user_id, b_matrix, collection):
  #Assuming right now users stored in user-array, may need to change this to accomadate grabbing it from the database
  currUser = users[int(user_id)]
  recs = currUser.get_books(b_matrix)
  recList = []
  for book_id, _ in recs:
    title, author, url = get_book_data(book_id, collection)
    if title != title:
      recList.append({"id":book_id, "name":"","author":author,"url":url})
    else:
      recList.append({"id":book_id, "name":title,"author":author,"url":url})
  return recList

def update_model(users, user_id, init_flag, sentiments, ratings, model_matrix, b_matrix, model): #sentiments is (book_id,sentiment)
  print(init_flag)
  print(type(user_id))
  returnBool = False
  if init_flag:
    ratings.append([sentiments[0],5*sentiments[1]])
    print(len(ratings)) 
    if len(ratings) == 18:
      returnBool = True
      users[int(user_id)] = User(ratings, model_matrix, int(user_id), 24, model)
      users.append(0)
      ratings = []
  else:
    improve(sentiments[1], sentiments[0], users[int(user_id)], b_matrix) 
  return returnBool