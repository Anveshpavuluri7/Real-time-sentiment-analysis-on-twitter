from flask import Flask,request,render_template,url_for
import snscrape.modules.twitter as sntwitter
import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib
import base64
import os
matplotlib.use('Agg')

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
#model_max_length = 256

labels = ['Negative', 'Neutral', 'Positive']

def make_query(data_from_user):
    if(data_from_user[0]=="option1"):
        res=f"(#{data_from_user[1]}) until:{data_from_user[4]} since:{data_from_user[3]}"
    elif(data_from_user[0]=="option2"):
        res=f"(to:{data_from_user[1]}) until:{data_from_user[4]} since:{data_from_user[3]}"  
    elif(data_from_user[0]=="option3"):
        res=f"(from:{data_from_user[1]}) until:{data_from_user[4]} since:{data_from_user[3]}"  
    else:
        res=f"(@{data_from_user[1]}) until:{data_from_user[4]} since:{data_from_user[3]}"  
    return res

def process_query(query,limit):
    # query = "(from:elonmusk) until:2023-02-19 since:2020-01-01"
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        # print(vars(tweet))
        # break
        if(len(tweets) == limit):
            break
        else:
            tweets.append([tweet.date,tweet.user.username,tweet.rawContent])
    df = pd.DataFrame(tweets,columns=["Date","User","Tweet"])
    # df.to_csv("./static/tweets.csv",encoding="utf-8")
    # # print(df)
    # with open('./static/tweets.csv','r') as csvfile:
    #     reader =  csv.reader(csvfile)
    #     tweet = []
    #     for row in reader:
    #         tweet.append(row[2])
    tweet=df['Tweet'].tolist()
    print(len(tweet))
    return tweet

def preprocess(tweet):
    # precprcess tweet
    tweet_words = []
    list_words=tweet.split(' ')
    # count=0
    for word in list_words:
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
        # count+=1
        # if(count==200):
        #     break
    tweet_proc = " ".join(tweet_words)
    return tweet_proc

def predict(tweet,result):
    tweet=preprocess(tweet)
    # sentiment analysis
    encoded_tweet = tokenizer(tweet,max_length=256,truncation=True, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    maxValue=-1
    maxIndex=-1
    for i in range(len(scores)):
        if(maxValue<scores[i]):
            maxIndex=i
            maxValue=scores[i]
        # l = labels[i]
        # s = scores[i]
        # print(l,s)
    if(maxIndex==0):#negative
        result[0]+=1
    elif(maxIndex==1):#neutral
        result[1]+=1
    else:#positive
        result[2]+=1
    return

def create_bar_plot(data,filename):
    plt.bar(range(len(data)), data)
    plt.xlabel('Negative ---- Neutral ---- Positive')
    plt.ylabel('Count of Tweets')
    plt.title('Bar Plot')
    plt.grid(False)
    plt.savefig(filename, format='png')
    # Encode the chart image to base64
    with open(filename, 'rb') as file:
        bar_plot_image = base64.b64encode(file.read()).decode('utf-8')
    return bar_plot_image

def create_pie_chart(data, filename):
    # Create the pie chart
    plt.pie(data, labels={"Negative", "Neutral",
            "Positive"}, autopct='%1.1f%%')
    plt.axis('equal')

    # Save the chart to a file
    plt.savefig(filename, format='png')

    # Encode the chart image to base64
    with open(filename, 'rb') as file:
        chart_image = base64.b64encode(file.read()).decode('utf-8')
    return chart_image


app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def result():
        # Handle the form submission
        form_data = request.form
        data_from_user=[]
        # print(form_data)
        data_from_user.append(form_data["selection"])#0
        data_from_user.append(form_data["topic"])#1
        data_from_user.append(form_data["limit"])#2
        data_from_user.append(form_data["sDate"])#3
        data_from_user.append(form_data["eDate"])#4
        # print(data_from_user)
        query=make_query(data_from_user)
        print("Query    ",query)
        limit=int(data_from_user[2])
        tweets=process_query(query,limit)
        result=[0,0,0]
        for tweet in tweets:
            predict(tweet,result)
        # print(result)
        filename_pie = './static/pie_chart.png'
        filename_bar = './static/bar_plot.png'
        if os.path.exists(filename_bar):
            os.remove(filename_bar)
        bar_plot=create_bar_plot(result,filename_bar)
        #chart_image = create_pie_chart(result, filename_pie)
        return render_template('result.html', count=limit, positive=result[2], neutral=result[1], negative=result[0])



if __name__=='__main__':
    app.run(debug=True)
