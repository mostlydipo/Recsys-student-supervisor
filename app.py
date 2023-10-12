# Core pkg
import streamlit as st
import streamlit.components.v1 as stc
import re
from wordcloud import WordCloud

# Additional required packages
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load EDA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(data):
    df = pd.read_json(data)
    df['Research'] = df['Research'].astype(str)
    df['Expertise'] = df['Expertise'].astype(str)
    df['combined'] = df['Research'] + ' ' + df['Expertise']
    return df


# Vectorize + cosine similarity matrix
def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

# Modify this function to be case insensitive and handle whitespaces
def get_index_from_title(title, df):
    title = title.lower().strip()
    return df[df['combined'].apply(lambda x: bool(re.search(title, x.lower())))].index[0]

# Recommendation system
@st.cache_resource
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=5):
    idx = get_index_from_title(title, df)
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Ensure the number of recommendations doesn't exceed available data
    num_of_rec = min(num_of_rec, len(sim_scores)-1)

    selected_supervisor_indices = [i[0] for i in sim_scores[1:num_of_rec+1]]
    selected_supervisor_scores = [i[1] for i in sim_scores[1:num_of_rec+1]]
    result_df = df.iloc[selected_supervisor_indices]
    result_df['similarity_score'] = selected_supervisor_scores
    final_recommended_supervisor = result_df[['Name','similarity_score','Profile', 'Email']]
    return final_recommended_supervisor

# Modify this function to be case insensitive and handle whitespaces
@st.cache_resource
def search_term_if_not_found(term, df):
    term = term.lower().strip()
    result_df = df[df['Name'].str.contains(term, case=False)]
    return result_df


#create wordcloud
def display_word_cloud(data, title):
    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(' '.join(data))
    
    fig, ax = plt.subplots(figsize=(15,8))
    ax.imshow(wordcloud)
    ax.axis('off')
    ax.set_title(title)  # Setting a title for the word cloud
    
    st.pyplot(fig)  # passing explicit figure to st.pyplot


def score_to_percentage(score):
    return round(score * 100)



# Display the star rating with star symbols
#def display_star_rating(star_rating):
    #full_star = "★"
    #half_star = "☆"
    #stars = full_star * int(star_rating)
    #if star_rating % 1:
        #stars += half_star
    #return stars



# CSS style
RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}%</p>
<p style="color:blue;"><span style="color:black;">🔗Profile:</span><a href="{}" target="_blank">View Profile</a></p>
<p style="color:blue;"><span style="color:black;"> ✉ Mail:</span>{}</p>
<p style="color:blue;"><span style="color:black;"> 📖 Publications:</span><br> {}</p>
</div>
"""




# Feedback storage
feedback_data = []

def main():
    st.title("Find me a Supervisor")

    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    df = load_data("data/newprofessor_data.json")
    search_fields = ["Combined", "Research", "Expertise"]
    search_field_selected = st.sidebar.radio("Search By", search_fields)

    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(14))

        # Button to display word cloud for Expertise
        if st.button("Display Word Cloud for Expertise"):
            display_word_cloud(df['Expertise'], "Expertise Word Cloud")
    
        # Button to display word cloud for Research
        if st.button("Display Word Cloud for Research"):
            display_word_cloud(df['Research'], "Research Word Cloud")
        #if st.button("Display Word Cloud for Expertise"):
            #display_word_cloud(df['combined'])

    elif choice == "Recommend":
        st.subheader("Recommend Supervisor")
        
        if search_field_selected == "Research":
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Research'])
        elif search_field_selected == "Expertise":
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['Expertise'])
        else:
            cosine_sim_mat = vectorize_text_to_cosine_mat(df['combined'])

        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number of Recommendations", min_value=2, max_value=10, value=3)
        
        if search_term:
            try:
                result = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                for row in result.iterrows():
                    rec_supervisor = row[1]['Name']
                    rec_score = score_to_percentage(row[1]['similarity_score'])
                    rec_profile = row[1]['Profile']
                    rec_mail = row[1]['Email']
                    publications_data = df[df['Name'] == rec_supervisor]['Publications'].values

                    # Prepare the publications data
                    if publications_data:
                        if isinstance(publications_data[0], list):
                            # Create clickable Google search links for each publication
                            publications = publications_data[0]
                            rec_publications = ', '.join([f'<a href="https://www.google.com/search?q={pub}" target="_blank">{pub}</a>' for pub in publications])
                        elif isinstance(publications_data[0], str):
                            pub = publications_data[0]
                            rec_publications = f'<a href="https://www.google.com/search?q={pub}" target="_blank">{pub}</a>'
                        else:
                            rec_publications = "No publications listed"
                    else:
                        rec_publications = "No publications listed"

                    # Display the recommendation using the updated template
                    stc.html(RESULT_TEMP.format(rec_supervisor, f"{rec_score}", rec_profile, rec_mail, rec_publications), height=350)

                

                # Feedback loop
                feedback = st.slider("Rate the quality of recommendations (1 being least relevant, 5 being most relevant)", 1, 5)
                if st.button("Submit Feedback"):
                    feedback_data.append(feedback)
                    st.success("Thanks for your feedback!")
                    
            except Exception as e:
                st.warning(f"Error: {e}")
                st.info("Suggested Options include")
                result_df = search_term_if_not_found(search_term, df)
                st.dataframe(result_df)

    else:
        st.subheader("About")
        st.text("Built with Streamlit, Pandas and more!")

if __name__ == '__main__':
    main()