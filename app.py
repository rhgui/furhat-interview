import streamlit as st
from furhat_robot import FurhatRobot
from cv_parser import CVParser
from question_generator import generate_questions


st.set_page_config(page_title="Furhat CV Interview", layout="centered")

st.title("Furhat Interview")

uploaded_file = st.file_uploader("Choose CV PDF", type="pdf")

if uploaded_file is not None:
    # save uploaded file to disk
    temp_path = "temp_cv.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    parser = CVParser()
    cv_data = parser.parse_cv(temp_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CV Analysis")
        st.json(cv_data)


        # wip, interview starts here
        if st.button("Start Interview", type="primary"):
            name = cv_data.get("name", "candidate")
            skills = cv_data.get("skills", [])
            skills_str = ", ".join(skills[:3]) if skills else "various skills"
            message = f"Hello {name}. I see you have experience with {skills_str}."

            with st.spinner("Connecting to Furhat..."):
                robot = FurhatRobot()
                try:
                    robot.connect()
                    robot.execute_sequence(message)
                    st.success("Interview sequence completed!")
                finally:
                    robot.disconnect()

    with col2:
        st.subheader("Summary")
        if "error" not in cv_data:
            st.write(f"**Name:** {cv_data.get('name', 'Unknown')}")
            st.write(f"**Skills:** {', '.join(cv_data.get('skills', []))}")
            st.write("Ready for technical interview!")
        else:
            st.error(cv_data["error"])
