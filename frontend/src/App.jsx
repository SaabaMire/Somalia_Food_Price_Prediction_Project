import React from 'react';
import PredictionForm from './PredictionForm';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      
      {/* HEADER BAR */}
      <header className="bg-teal-700 shadow-md">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-white tracking-wide">
            Somalia Food Price Predictor 
          </h1>
          <span className="text-sm font-medium text-teal-200">
            Powered by GBR Machine Learning
          </span>
        </div>
      </header>

      {/* MAIN CONTENT AREA */}
      <main className="max-w-7xl mx-auto py-8 sm:py-12 px-4 sm:px-6 lg:px-8">
        <PredictionForm />
      </main>

      {/* FOOTER */}
      <footer className="w-full py-4 text-center text-sm text-gray-500 border-t mt-12">
        &copy; 2025 ML Deployment Project. All Rights Reserved SaabaMire.
      </footer>
    </div>
  );
}

export default App;