import React, { useState } from 'react';
import './App.css';
import KawaiiRobot from './components/KawaiiRobot';

function App() {
  const [context, setContext] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!context || !question) return;

    setLoading(true);
    setAnswer(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context, question }),
      });
      
      const data = await response.json();
      setAnswer(data.answer);
    } catch (error) {
      console.error("Erreur:", error);
      setAnswer("Oups ! J'ai eu un bug...");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setContext('');
    setQuestion('');
    setAnswer(null);
  };

  return (
    <div className="app-container">
      <h1 className="title">Optimus Ask</h1>
      
      <KawaiiRobot isThinking={loading} />

      <div className="card">
        {!answer ? (
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Le Contexte :</label>
              <textarea 
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="Colle ici le texte dans lequel je dois chercher..."
                rows="5"
                disabled={loading}
              />
            </div>
            <div className="form-group">
              <label>Ta Question :</label>
              <input 
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Que veux-tu savoir ?"
                disabled={loading}
              />
            </div>
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? 'Recherche...' : 'Demander au Bot'}
            </button>
          </form>
        ) : (
          <div className="result-container">
            <h3>Voici ma réponse :</h3>
            <p className="answer-box">"{answer}"</p>
            <button onClick={handleReset} className="btn-secondary">
              Nouvelle Recherche ↺
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;