import pandas as pd
import numpy as np
import requests
import tkinter as tk
from tkinter import ttk
from tkinter import *
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Import the data
ratings = pd.read_csv('ratingsData.csv', header = 0)
df_ratings = pd.read_csv('ratingsData.csv', header = 0, usecols = ['user_id', 'book_id', 'rating'],
                         dtype = {'user_id': 'int32', 'book_id': 'int32', 'rating': 'float32'})

books_og = pd.read_csv('booksInfo.csv', header = 0)
df_books = pd.read_csv('booksInfo.csv', header = 0, usecols = ['book_id', 'title'],
                       dtype = {'book_id': 'int32', 'title': 'str'})

# View number of users and books in the ratings data
num_users = len(df_ratings.user_id.unique())
num_items = len(df_ratings.book_id.unique())

# Not every user can review every book so there are lots of 0 counts
total_cnt = num_users * num_items
rating_zero_cnt = total_cnt - df_ratings.shape[0]

# Count of each rating
df_ratings_cnt_tmp = pd.DataFrame(df_ratings.groupby('rating').size(), columns = ['count'])

# Append counts of 0 rating to df_ratings_cnt
df_ratings_cnt = df_ratings_cnt_tmp.append(
    pd.DataFrame({'count': rating_zero_cnt}, index = [0]),
    verify_integrity = True,
).sort_index()

# count for 0 rating score is too big to compare with others so we log transform for count values
df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])

# Rating frequency of books
df_books_cnt = pd.DataFrame(df_ratings.groupby('book_id').size(), columns = ['count'])

# Remove books that were rated less than 50 times
popularity_thresh = 50
popular_books = list(set(df_books_cnt.query('count >= @popularity_thresh').index))
df_ratings_drop_books = df_ratings[df_ratings.book_id.isin(popular_books)]

# How many ratings does each person do
df_users_cnt = pd.DataFrame(df_ratings_drop_books.groupby('user_id').size(), columns = ['count'])

# Filter out the users who reviewed less than 3 times
ratings_thresh = 3
active_users = list(set(df_users_cnt.query('count >= @ratings_thresh').index))
df_ratings_drop_users = df_ratings_drop_books[df_ratings_drop_books.user_id.isin(active_users)]

# Keep 3 columns of the new ratings dataframe where we dropped the less popular books and people who
# didnt rate much
df_ratings_drop_users = df_ratings_drop_users[['book_id', 'user_id', 'rating']]

# New dataframe of merged ratings dataframe and the book df with the book title
merged = pd.merge(df_ratings_drop_users, df_books, on = 'book_id', how = 'left')
merged2 = merged.dropna()

# Keep only the title, user_id and rating columns from the merged dataframe
df = merged2[['title','user_id','rating']]

# create a list of all the book titles to put in our app as selections
book_titles = df['title'].unique()
book_titles = list(book_titles)

# Create pivot table with book_id as rows and user_id as the column and rating as the values
merged_drop_users = df.pivot_table(index = 'title', columns = 'user_id',
                                  values = 'rating', aggfunc = 'mean', fill_value = 0)

# Create the csr matrix for the book, user, rating matrix
user_book_table_matrix = csr_matrix(merged_drop_users.values)
# Define the model
knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 20, n_jobs = 1)
# fit the model
knn.fit(user_book_table_matrix)

# the UI
root = Tk()
root.title("Book Recommendation System")
root.geometry('650x850')

l = Label(root, text = "Book Recommendations", font = ('Arial', 16))
l.grid(row = 0, column = 1)

q2 = Label(root, text = "How many recommendations do you want?")
q2.grid(row = 1, column = 1)

# Dropdown for number of recomendations the user wants
number_of_recs_chose = IntVar(root)
number_of_recs_chose.set(5) #default value

# Add the number of recommendations button to the window
q3 = OptionMenu(root, number_of_recs_chose, 1,2,3,4,5,6,7,8,9,10)
q3.grid(row = 2, column = 1)

q1 = Label(root, text = "What book did you love?")
q1.grid(row = 3, column = 1)

def buttonFunction():
    myLabel = Label(root, text = book.get())
    myLabel.grid()

def remove_text():
    output_label.destroy()
    run_it_button['state'] = NORMAL
    print(run_it_button.winfo_exists())
    output_label2.destroy()
    run_it_button['state'] = NORMAL
    print(run_it_button.winfo_exists())

def book_recs():
    # output labels are global variables so they can be used everywhere
    global output_label
    global output_label2

    # Create book and distance lists for when we want to display the top 10
    book_rec_list = []
    distance = []

    num_recs = number_of_recs_chose.get()
    choice = book.get()

    # Find the book index of the book that the user selected as their favourite
    book_index = np.where(merged_drop_users.index == choice)

    # Find the number index of corresponding to the book index
    query_index = book_index[0]

    # Back to the title of the book from the index. Just doing this to get the title directly from the table
    # but since the user writes the title of the book they want we could just use that
    chosen_book_with_index = merged_drop_users.index[query_index][0]

    # Calculate the distances and index of where the book is located to calculate the closest other books
    distances, indices = knn.kneighbors(merged_drop_users.iloc[query_index, :].values.reshape(1, -1),
                                        n_neighbors=num_recs + 1)  # n_neighbors = number of recommendations we want

    # For all the books, add to the lists of book, distance so we can rank them and take top 10
    for i in range(0, len(distances.flatten())):
        if i != 0:
            book_rec_list.append(merged_drop_users.index[indices.flatten()[i]])
            distance.append(distances.flatten()[i])

    # Create a 1-dimension array with book and distance lists
    m = pd.Series(book_rec_list, name='book')
    d = pd.Series(distance, name='distance')

    # create concatenated array with both arrays above
    recommend = pd.concat([m, d], axis=1)

    # sort the values based on the distance
    recommend = recommend.sort_values('distance', ascending=False)

    # Label for the intro to the recommendations
    output_label = Label(root, text = 'Recommendations for {0}:\n'.format(chosen_book_with_index),
                         font = ('Arial', 16), anchor = 'center')
    output_label.grid(row = 9, column = 1)

    # Print the recommendation books
    var = StringVar()
    display_list_sep = '\n'
    book_rec_list = recommend['book'].tolist()
    display_list = display_list_sep.join(book_rec_list)
    output_label2 = Label(root, textvariable=var, anchor = 'center')
    var.set(display_list)
    output_label2.grid(row=10, column=1)

    run_it_button['state'] = DISABLED

# update the listbox
def update(data):
    # Clear the listbox
    all_books.delete(0, END)
    # Add books to listbox
    for item in data:
        all_books.insert(END,item)

# Update entry box with listbox clicked
def fillout(e):
    # Delete whatever is in the entry box
    book.delete(0,END)
    # Add clicked list in item to entry box
    book.insert(0,all_books.get(ACTIVE))

def check(e):
    # grab what was typed
    typed = book.get()

    if typed == '':
        data = book_titles
    else:
        data = []
        for item in book_titles:
            if typed.lower() in item.lower():
                data.append(item)
    # update our listbox with selected items
    update(data)

# fill in box
book = Entry(root, width = 60)
book.grid(row = 4, column = 1,  padx = 45)
# book.pack()

# List box
all_books = Listbox(root, width = 60)
all_books.grid(row = 5, column = 1,  padx = 45)
# all_books.pack()

# List of books
update(book_titles)

# Create a binding on the listbox onclick
all_books.bind("<<ListboxSelect>>", fillout)

# Create a binding on the entry box
book.bind("<KeyRelease>", check)

run_it_button = Button(root, text = "Run It", command = book_recs)
run_it_button.grid(row=6, column=1)
# run_it_button.pack()

clear_button = Button(root, text = "Clear Recommendations", command = remove_text)
clear_button.grid(row=7, column=1)
# clear_button.pack()

# space = Label(root, text = " ")
# space.pack()

root.mainloop()