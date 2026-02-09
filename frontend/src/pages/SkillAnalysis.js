import React from 'react';
import {useForm } from "react-hook-form"

const SkillAnalysis = () => {
  const { register , handleSubmit} = useForm();
  
  const onSubmit = (data) => {

  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <label>Enter your skills: <br />
                <input type="text" {...register("skills")}/>
            <br />
            </label>
            <label>Enter your experience: <br />
                <input type="number" {...register("experience_years")}/>
            <br />
            </label>
            <label>Enter your salary expectations: <br />
                <input type="number" {...register("expected_salary")}/>
            <br />
            </label>
            <label>Enter your preferred location: <br />
                <input type="text" {...register("preferred_location")}/>
            <br />
            </label>
            <button type="submit">Submit</button>
    </form>
  );
};

export default SkillAnalysis;