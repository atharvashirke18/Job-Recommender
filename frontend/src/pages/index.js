import React, { useState } from "react";
import '../Navbar.css';

function MyForm() {
    const [formData , setFormData] = useState({
        skills:'',
        experience_years: '',
        expected_salary: '',
        preferred_location: '',
        top_n: '10'
    });

    const handleChange = (e) => {
        const { name , value } = e.target;
        
        setFormData( prevState =>({
            ...prevState,
            [name]:value
        }));
    };
    
    const handleSubmit = (e) =>{
        e.preventDefault();

        const {skills , experience_years , expected_salary , preferred_location , top_n} = formData;
        console.log("Form Submitted", {skills, experience_years ,expected_salary, preferred_location, top_n});
    }

     return (
        <form onSubmit={handleSubmit}>
            <label>Enter your skills: <br />
                <input type="text" name="skills" value={formData.skills} onChange={handleChange}/>
            <br />
            </label>
            <label>Enter your experience: <br />
                <input type="text" name="experience_years" value={formData.experience_years} onChange={handleChange}/>
            <br />
            </label>
            <label>Enter your salary expectations: <br />
                <input type="text" name="expected_salary" value={formData.expected_salary} onChange={handleChange}/>
            <br />
            </label>
            <label>Enter your preferred location: <br />
                <input type="text" name="preferred_location" value={formData.preferred_location} onChange={handleChange}/>
            <br />
            </label>
            <button type="submit">Submit</button>
        </form>
     );
}

const Home = () => {
    return (
        <div style={{
                display: "flex",
                justifyContent: "centre",
                alignItems: "centre",
                height: "100vh",
            }}
        >
            <MyForm />
        </div>
    );
}

export default Home;
