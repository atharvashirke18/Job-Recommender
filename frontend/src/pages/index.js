import React, { useState } from "react";
import '../Navbar.css';
import MyForm from '../components/Form';
import JobReco from "./JobReco";

const Home = () => {
    const [isSubmitted, setIsSubmitted] = useState(false);
    return (
        <div style={{
                display: "flex",
                justifyContent: "centre",
                alignItems: "centre",
                height: "100vh",
            }}
        >
            <div>
             <MyForm isSubmitted={false}/> 
             </div>
        </div>
    );
}

export default Home;