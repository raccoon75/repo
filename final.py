import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import altair as alt

with st.echo(code_location='below'):

    def print_hello(name = "Гость"):
        st.write(f"## Привет, {name}!")

    name = st.text_input("Ваше имя", key='name', value= "Гость")
    print_hello(name)

    st.write(f"## Spotify top chart 2010 - 2019")

    """
    Здесь собраны треки, входившие в топ чарт Спотифая за указанные годы. 
    В датасете также указаны разные параметры для каждого трека: 
    насколько он громкий, энергичный, с какой частотой битов в минуту, 
    насколько под него можно танцевать, насколько он популярен, и тд. 
    Посмотрим на него поближе:)
    """
    @st.cache
    def get_data():
        data_url = "https://raw.githubusercontent.com/raccoon75/repo1/main/Spotify%202010%20-%202019%20Top%20100.csv"
        return pd.read_csv(data_url)

    df = get_data()
    df.rename(columns ={"title": "Song title", 'artist': "Author", "top genre":"Song genre", "year released":"Year released", "added":"Added to Spotify", "bpm":"Beats per min", "nrgy":"How energetic", "dnce":"How danceable", "dB":"How loud", "live":"How likely song is live", "val":"How positive", "dur":"Duration", "acous":"How acoustic", "spch":"How much words matter", "pop":"Popularity", "top year":"Became hit", "artist type": "Number of singers"}, inplace=True)
    df.dropna().drop(459, axis = 0)
    df


    """
    Здесь 444 исполнителя и 1000 треков. Посмотрим, кто из исполнителей самые продуктивные) Ниже представлен топ 20 исполнителей по количеству треков.
    """
    auhtors = df["Author"].unique()
    dictionary = {}
    for author in auhtors:
        q = df[df['Author']==author]["Song title"].count()
        dictionary[author]=q
    author_song = pd.DataFrame(list(dictionary.items())).rename(columns = {0:"Authors",1:"Number of songs"}).sort_values(by="Number of songs",ascending=False)

    top10 = author_song[:20]
    number_of_songs = list(top10['Number of songs'])
    artists = tuple(top10['Authors'])
    y_pos1 = np.arange(len(artists))
    fig, ax = plt.subplots()
    ax = plt.barh(y_pos1, number_of_songs, color = [plt.cm.Spectral(i/float(len(top10))) for i in range(len(top10))], edgecolor = "black")
    plt.yticks(y_pos1, artists)
    plt.title("Количество песен исполнителя, топ 20")
    st.pyplot(fig)

    """А теперь посмотрим, какие жанры представлены в топ чарте"""

    genres = df["Song genre"].unique()
    dictionary1 = {}
    keys1=[]
    size_of_groups =[]
    for genre in genres:
        q = df[df['Song genre']==genre]["Song title"].count()
        keys1.append(genre)
        dictionary1[genre]=q
        size_of_groups.append(q)
    genre_song = pd.DataFrame(list(dictionary1.items())).rename(columns = {0:"Genre",1:"Number of songs"}).sort_values(by="Number of songs",ascending=False)

    top10genres = list(genre_song[:50]['Number of songs'])
    rest = []
    for i in top10genres:
        if top10genres[0]/i>30.08:
            rest.append(i)
    length = len(top10genres)-len(rest)
    listforchart = top10genres[:length]
    listforchart.append(sum(rest))
    data_names = list(genre_song["Genre"])[:length]
    data_names.append('other')
    totall = sum(listforchart)
    labels = [f"{n} ({v/totall:.1%})" for n,v in zip(data_names, listforchart)]
    colors = [plt.cm.Spectral(i/float(len(listforchart))) for i in range(len(listforchart))]
    fig, ax = plt.subplots()
    ax = plt.pie(listforchart,radius = 1.5, colors=colors)
    plt.legend(
        bbox_to_anchor = (-0.4, 0.8, 0.25, 0.25),
        loc = 'best', labels = labels)
    plt.title(label = "Структура жанров в топ чарте",fontsize=15,pad = 35)
    st.pyplot(fig)

    """
    Как уже упоминалось, каждый трек оценен по разным критериям. Вот их список, если что:
    """
    st.write('Beats per min, ','How energetic, ', 'How danceable, ', 'How loud, ', 'How likely song is live, ',
             'How positive, ', 'Duration, ', 'How acoustic, ','How much words matter')

    dfcharacteristics = pd.DataFrame()
    dfcharacteristics['Beats per min'] = df['Beats per min']
    dfcharacteristics['How energetic'] = df['How energetic']
    dfcharacteristics['How danceable'] = df['How danceable']
    dfcharacteristics['How loud'] = df['How loud']
    dfcharacteristics['How likely song is live'] = df['How likely song is live']
    dfcharacteristics['How positive'] = df['How positive']
    dfcharacteristics['Duration'] = df['Duration']
    dfcharacteristics['How acoustic'] = df['How acoustic']
    dfcharacteristics['How much words matter'] = df['How much words matter']


    """ Попробуем посмотреть, как коррелируют друг с другом эти факторы"""


    fig, ax = plt.subplots()
    ax = sns.heatmap(dfcharacteristics.corr(), xticklabels=dfcharacteristics.corr().columns, yticklabels=dfcharacteristics.corr().columns, cmap='OrRd',center=0, annot=True)
    plt.figure(figsize=(8,7), dpi= 80)
    plt.title('Корреляция характеристик трека', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(fig)

    """ По крайней мере, теперь знаем, что чем музыка громче, тем она энергичнее, 
    а вот на сколько под музыку можно танцевать по энергичности не скажешь:)"""

    """
    А теперь выясним, как коррелируют эти характеристики с популярностью трека
    """
    source = dfcharacteristics
    source['Popularity'] = df['Popularity']

    base = alt.Chart(source, width=170, height=170).mark_point(filled=True).encode(
        x=alt.X(alt.repeat("column"), type='quantitative'),
        y=alt.Y('Popularity', type='quantitative'),
        tooltip="Title:N"
    ).repeat(column=['How loud', 'How energetic', 'How danceable', 'Beats per min']).interactive()

    base

    base1 = alt.Chart(source, width=170, height=170).mark_point(filled=True).encode(
        x=alt.X(alt.repeat("column"), type='quantitative'),
        y=alt.Y('Popularity', type='quantitative'),
        tooltip="Title:N"
    ).repeat(column=['How positive','How likely song is live', 'Duration', 'How acoustic']).interactive()

    base1

    """
    Осталось посмотреть на конкретные треки) Выбери исполнителя:
    """
    artistt = st.selectbox(
            "Artist", df["Author"].value_counts().index
        )
    df_selection = df[lambda x: x["Author"] == artistt]
    df_selection

    """
    И выбери его/ее трек:
    """
    songg = st.selectbox("Song", df_selection['Song title'].value_counts().index )
    df_selection2 = df_selection[lambda x: x["Song title"] == songg]
    df_selection2

    """
    Посмотрим, какие показатели у этого трека:
    """

    dfsong = pd.DataFrame()
    dfsong['Song title'] = df['Song title']
    dfsong['How energetic'] = df['How energetic']
    dfsong['How danceable'] = df['How danceable']
    dfsong['How loud'] = df['How loud']
    dfsong['How likely song is live'] = df['How likely song is live']
    dfsong['How positive'] = df['How positive']
    dfsong['How acoustic'] = df['How acoustic']
    dfsong['How much words matter'] = df['How much words matter']
    criteria = list(dfsong)[1:]
    values = dfsong[df['Song title'] == songg].values[0][1:8]
    new_frame = pd.DataFrame({'Characteristics': criteria,
                               'PercentageOf100': values})

    chart = alt.Chart(new_frame).transform_calculate(
        PercentOfTotal="datum.PercentageOf100"
    ).mark_bar().encode(
        x=alt.X('PercentageOf100'),
        y='Characteristics:N',
    ).configure_mark(
        opacity=0.7,
        color='purple'
    ).properties(
        width=500,
        height=300
    )
    chart

    """
    Посмотрим, как быстро трек стал хитом: когда был релиз, а когда попал в список хитов
    """

    ### FROM: Код основан на 18 пункте отсюда: https://habr.com/ru/post/468295/  (но поменяла под себя)
    left_label = [str(c) + ', '+ str(round(y)) for c, y in zip(df_selection2["Song title"], df_selection2['Year released'])]
    right_label = [str(c) + ', '+ str(round(y)) for c, y in zip(df_selection2["Song title"], df_selection2['Became hit'])]

    def newline(p1, p2, color='black'):
        ax = plt.gca()
        l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='purple', marker='o', markersize=6)
        ax.add_line(l)
        return l


    fig, ax = plt.subplots(1,1,figsize=(8,6), dpi= 80)
    ax.vlines(x=1, ymin=1974, ymax=2024, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
    ax.vlines(x=3, ymin=1974, ymax=2024, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

    ax.scatter(y=df_selection2['Year released'], x=np.repeat(1, df_selection2.shape[0]), s=10, color='black', alpha=0.7)
    ax.scatter(y=df_selection2['Became hit'], x=np.repeat(3, df_selection2.shape[0]), s=10, color='black', alpha=0.7)

    for p1, p2, c in zip(df_selection2['Year released'], df_selection2['Became hit'], df_selection2['Song title']):
        newline([1,p1], [3,p2])

    ax.text(1-0.05, 2024, 'Released', horizontalalignment='right', verticalalignment='center', fontdict={'size':12, 'weight':600})
    ax.text(3+0.05, 2024, 'Became hit', horizontalalignment='left', verticalalignment='center', fontdict={'size':12, 'weight':600})

    ax.set(xlim=(0,4), ylim=(1974,2024), ylabel='Years')
    ax.set_xticks([1,3])
    ax.set_xticklabels([" ", " "])
    plt.yticks(np.arange(1974, 2024, 3), fontsize=10)

    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.0)
    st.pyplot(fig)
### END FROM
    """
    Спасибо, что заглянул(-а)!
    """