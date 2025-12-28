import React, { useState } from 'react';
import './ResumeMatcher.css';

const ResumeMatcher = () => {
  const [resume, setResume] = useState('');
  const [jobDesc, setJobDesc] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resume,
          job_description: jobDesc
        })
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="resume-matcher">
      <h2>Resume-Job Match Predictor</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Resume Text:</label>
          <textarea 
            value={resume} 
            onChange={(e) => setResume(e.target.value)}
            rows={6}
            required
          />
        </div>
        
        <div>
          <label>Job Description:</label>
          <textarea 
            value={jobDesc} 
            onChange={(e) => setJobDesc(e.target.value)}
            rows={6}
            required
          />
        </div>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Check Match'}
        </button>
      </form>
      
      {result && (
        <div className="result">
          <h3>Match Percentage: {result.match_percentage}%</h3>
          <div className="progress-bar">
            <div 
              className="progress" 
              style={{ width: `${result.match_percentage}%` }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResumeMatcher;