import pdfplumber
import streamlit as st
import pickle
import re
import nltk
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

#
def getResult(JD_txt, resume_txt, tfidf_vectorizer):
    # Ensure that JD_txt and resume_txt are strings
    JD_txt = str(JD_txt)
    resume_txt = str(resume_txt)

    content = [JD_txt, resume_txt]

    matrix = tfidf_vectorizer.transform(content)

    # No need to convert to dense, cosine_similarity can handle sparse matrices
    similarity_matrix = cosine_similarity(matrix)

    match = similarity_matrix[0][1] * 100

    return match
# def getResult(JD_txt, resume_txt):
#     content = [JD_txt, resume_txt]

#     cv = CountVectorizer()

#     matrix = cv.fit_transform(content)

#     similarity_matrix = cosine_similarity(matrix)

#     match = similarity_matrix[0][1] * 100

#     return match
#

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

        job_title = category_name

        #

        uploadedJD = st.file_uploader("Upload Job Description", type="pdf")
        click = st.button("Process")

        try:
            global job_description
            with pdfplumber.open(uploadedJD) as pdf:
                pages = pdf.pages[0]
                job_description = pages.extract_text()

        except:
             st.write("")

        if click:
            match = getResult(job_description, uploaded_file)
            match = round(match, 2)
            st.write("Match Percentage: ", match, "%")
        #

        if st.button('Search Similar Profiles'):
            params = {
                "engine": "google_jobs",
                "q": job_title,
                "l": "bangalore",
                "ltype": "1",
                "hl": "en",
                "api_key": "0cd4e706e3406157cd953b170a8a0a1aa822ad137bdf7c8e51f58eac91a8c926"
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            jobs_results = results.get('jobs_results', [])

            st.subheader('Search Results')
            for job in jobs_results:
                job_title = job.get('title', 'N/A')
                job_link = job.get('related_links', 'N/A')
                st.write(f"[{job_title}]({job_link})")



# python main
if __name__ == "__main__":
    main()




# import streamlit as st
# import pickle
# import re
# import nltk
# from serpapi import GoogleSearch

# nltk.download('punkt')
# nltk.download('stopwords')

# #loading models
# clf = pickle.load(open('clf.pkl','rb'))
# tfidfd = pickle.load(open('tfidf.pkl','rb'))

# #
# def getResult(JD_txt, resume_txt):
#     content = [JD_txt, resume_txt]

#     cv = CountVectorizer()

#     matrix = cv.fit_transform(content)

#     similarity_matrix = cosine_similarity(matrix)

#     match = similarity_matrix[0][1] * 100

#     return match
# #

# def clean_resume(resume_text):
#     clean_text = re.sub('http\S+\s*', ' ', resume_text)
#     clean_text = re.sub('RT|cc', ' ', clean_text)
#     clean_text = re.sub('#\S+', '', clean_text)
#     clean_text = re.sub('@\S+', '  ', clean_text)
#     clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
#     clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
#     clean_text = re.sub('\s+', ' ', clean_text)
#     return clean_text
# # web app
# def main():
#     st.title("Resume Screening App")
#     uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

#     if uploaded_file is not None:
#         try:
#             resume_bytes = uploaded_file.read()
#             resume_text = resume_bytes.decode('utf-8')
#         except UnicodeDecodeError:
#             # If UTF-8 decoding fails, try decoding with 'latin-1'
#             resume_text = resume_bytes.decode('latin-1')

#         cleaned_resume = clean_resume(resume_text)
#         input_features = tfidfd.transform([cleaned_resume])
#         prediction_id = clf.predict(input_features)[0]
#         st.write(prediction_id)

#         # Map category ID to category name
#         category_mapping = {
#             15: "Java Developer",
#             23: "Testing",
#             8: "DevOps Engineer",
#             20: "Python Developer",
#             24: "Web Designing",
#             12: "HR",
#             13: "Hadoop",
#             3: "Blockchain",
#             10: "ETL Developer",
#             18: "Operations Manager",
#             6: "Data Science",
#             22: "Sales",
#             16: "Mechanical Engineer",
#             1: "Arts",
#             7: "Database",
#             11: "Electrical Engineering",
#             14: "Health and fitness",
#             19: "PMO",
#             4: "Business Analyst",
#             9: "DotNet Developer",
#             2: "Automation Testing",
#             17: "Network Security Engineer",
#             21: "SAP Developer",
#             5: "Civil Engineer",
#             0: "Advocate",
#         }

#         category_name = category_mapping.get(prediction_id, "Unknown")

#         st.write("Predicted Category:", category_name)

#         job_title = category_name

#         #
#         uploadedJD = st.file_uploader("Upload Job Description", type="pdf")
#         click = st.button("Process")

#         if click:
#              match = getResult(job_description, uploaded_file)
#              match = round(match, 2)
#              st.write("Match Percentage: ", match, "%")


#         try:
#             global job_description
#             with pdfplumber.open(uploadedJD) as pdf:
#                 pages = pdf.pages[0]
#                 job_description = pages.extract_text()

#         except:
#              st.write("")
#         #

#         if st.button('Search Similar Profiles'):
#             params = {
#                 "engine": "google_jobs",
#                 "q": job_title,
#                 "l": "bangalore",
#                 "ltype": "1",
#                 "hl": "en",
#                 "api_key": "0cd4e706e3406157cd953b170a8a0a1aa822ad137bdf7c8e51f58eac91a8c926"
#             }

#             search = GoogleSearch(params)
#             results = search.get_dict()

#             jobs_results = results.get('jobs_results', [])

#             st.subheader('Search Results')
#             for job in jobs_results:
#                 job_title = job.get('title', 'N/A')
#                 job_link = job.get('related_links', 'N/A')
#                 st.write(f"[{job_title}]({job_link})")



# # python main
# if __name__ == "__main__":
#     main()
