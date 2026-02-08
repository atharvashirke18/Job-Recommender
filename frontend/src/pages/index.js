import React from "react";
import '../Navbar.css';

const Home = () => {
    return (
        <div style={{
                display: "flex",
                justifyContent: "centre",
                alignItems: "centre",
                height: "100vh",
            }}
        >
            <h1>
                GeeksforGeeks is a Computer Science portal
                for geeks.
            </h1>
        </div>
    );
}

export default Home;
