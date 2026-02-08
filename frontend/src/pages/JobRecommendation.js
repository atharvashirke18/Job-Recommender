import React from 'react';

const languages = [
    { name: "JavaScript", founder: "Brendan Eich" },
    { name: "Python", founder: "Guido van Rossum" },
    { name: "Java", founder: "James Gosling" },
    { name: "C++", founder: "Bjarne Stroustrup" },
    { name: "Ruby", founder: "Yukihiro Matsumoto" },
];

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
                {/* {languages.map((language, index) => (
                    <li
                        key={index}
                        style={{
                            width:"100vw",
                            alignSelf: "center",
                            marginBottom: "10px",
                            padding: "20px",
                            backgroundColor: "#f4f4f4",
                            borderRadius: "4px",
                        }}
                    >
                        <strong>{language.name}</strong> - Founder:{" "}
                        {language.founder}
                    </li>
                ))} */}
                {listItems}
            </ul>
        </div>
    );
};

export default JobRecommendation;