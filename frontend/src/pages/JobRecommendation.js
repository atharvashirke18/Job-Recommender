import React from 'react';
import { FinalResponse } from '../components/Form';
import { formSubmitted } from '../components/Form';

const languages = [
    { name: "JavaScript", founder: "Brendan Eich" },
    { name: "Python", founder: "Guido van Rossum" },
    { name: "Java", founder: "James Gosling" },
    { name: "C++", founder: "Bjarne Stroustrup" },
    { name: "Ruby", founder: "Yukihiro Matsumoto" },
];


const reponse = FinalResponse;
console.log('Final Response from JobRecommendations.js: ', reponse);

const listItems = languages.map(key =>
    <li style={{
                display: "flex",
                flexWrap: "wrap",
                flex : "1 100%",
                whiteSpace: "nowrap",
                marginBottom: "10px",
                padding: "30px",
                backgroundColor: "#f4f4f4",
                borderRadius: "4px",
                        }}> 
        
        {key.name} : {key.founder} 
      
      </li>
);

const JobRecommendation = () => {
return (

            <div style={{ textAlign: "center" }}>
                <ul style={{padding:"10px" , display: "flex", flexWrap: "wrap"}}>
                    {listItems}
                </ul>
            </div>
    );
};

export default JobRecommendation;