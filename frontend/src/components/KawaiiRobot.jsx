import React from 'react';
import './KawaiiRobot.css';

const KawaiiRobot = ({ isThinking }) => {
  return (
    <div className={`robot-container ${isThinking ? 'thinking' : ''}`}>
      <div className="antenna">
        <div className="bulb"></div>
      </div>
      <div className="head">
        <div className="eyes">
          <div className="eye left"></div>
          <div className="eye right"></div>
        </div>
        <div className="mouth"></div>
      </div>
      <div className="body">
        <div className="screen">
            {isThinking ? (
                <div className="heartbeat">♥</div>
            ) : (
                <div className="status">OK</div>
            )}
        </div>
      </div>
      <div className="message">
        {isThinking ? "Hmm... Laisse-moi réfléchir..." : "Prêt pour ta question !"}
      </div>
    </div>
  );
};

export default KawaiiRobot;