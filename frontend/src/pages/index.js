import React, { useState } from "react";
import '../Navbar.css';

function MyForm() {
    const [formData , setFormData] = useState({
        skills:'',
        experience: '',
        salaryexp: ''
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

        const {skills , experience , salaryexp} = formData;
        console.log("Form Submitted", {skills, experience ,salaryexp});
    }

     return (
        <form onSubmit={handleSubmit}>
            <label>Enter your skills: <br />
                <input type="text" name="skills" value={formData.skills} onChange={handleChange}/>
            <br />
            </label>
            <label>Enter your experience: <br />
                <input type="text" name="experience" value={formData.experience} onChange={handleChange}/>
            <br />
            </label>
            <label>Enter your salaryexp: <br />
                <input type="text" name="salaryexp" value={formData.salaryexp} onChange={handleChange}/>
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
