import React, { useState } from "react";
import ThreeViewer from "./components/ThreeViewer.jsx";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [gltfUrl, setGltfUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (text.trim() === "") return;
    setLoading(true);
    setError(null);
    try {
      // Using the full URL for the backend
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to generate model: ${errorText}`);
      }

      // The response is directly the GLB file - create a blob URL
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setGltfUrl(url);
    } catch (err) {
      console.error("Error:", err);
      setError(
        `Failed to generate model. Please try again. Error: ${err.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-4xl font-bold mb-8 text-gray-800">
        Text-to-3D Generator
      </h1>
      <div className="w-full max-w-md bg-white p-6 rounded-lg shadow-md">
        <label
          className="block text-gray-700 text-sm font-bold mb-2"
          htmlFor="text-input"
        >
          Enter a description (e.g., 'a red car')
        </label>
        <input
          id="text-input"
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline mb-4"
          placeholder="Type your description here"
        />
        <button
          onClick={handleGenerate}
          disabled={loading || text.trim() === ""}
          className={`w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline ${
            loading || text.trim() === "" ? "opacity-50 cursor-not-allowed" : ""
          }`}
        >
          {loading ? "Generating..." : "Generate"}
        </button>
        {error && <p className="text-red-500 mt-4">{error}</p>}
      </div>
      <div className="mt-8 w-full max-w-4xl">
        {loading ? (
          <div className="flex justify-center items-center">
            <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : gltfUrl ? (
          <div className="flex flex-col items-center">
            <ThreeViewer gltfUrl={gltfUrl} />
            <a
              href={gltfUrl}
              download="generated_model.glb" // Updated to .glb
              className="mt-4 bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
            >
              Download GLB
            </a>
          </div>
        ) : (
          <p className="text-gray-600 text-center">
            Enter text and click Generate to see the 3D model.
          </p>
        )}
      </div>
    </div>
  );
}

export default App;
