"use client";

import React, { useEffect, useRef, useState } from "react";
import ResultWithSources from "./components/ResultWithSources";
import PromptBox from "./components/PromptBox";
import Button from "./components/Button";
import PageHeader from "./components/PageHeader";
import Title from "./components/Title";
import TwoColumnLayout from "./components/TwoColumnLayout";
import ButtonContainer from "./components/ButtonContainer";
import { useFilePicker } from "use-file-picker";
import axios from "axios";
import "./globals.css";

// This functional component is responsible for loading PDFs
const PDFLoader = () => {
  // Managing prompt, messages, and error states with useState
  const [prompt, setPrompt] = useState("How does XXX work?");
  const [messages, setMessages] = useState([
    {
      text: "Hi, I'm your AI assistant. What would you like to know?",
      type: "bot",
    },
  ]);
  const [error, setError] = useState("");

  const { openFilePicker, filesContent, isFilePickerLoading } = useFilePicker(
    {}
  );

  useEffect(() => {
    if (filesContent) {
      console.log("filesContent", filesContent);
    }
  }, [filesContent]);

  // This function updates the prompt value when the user types in the prompt box
  const handlePromptChange = (e) => {
    setPrompt(e.target.value);
  };

  // This function handles the submission of the form when the user hits 'Enter' or 'Submit'
  // It sends a GET request to the provided endpoint with the current prompt as the query
  const handleSubmit = async (endpoint) => {
    try {
      console.log(`sending ${prompt}`);
      console.log(`using ${endpoint}`);

      // A GET request is sent to the backend
      const response = await fetch(`/api/${endpoint}`, {
        method: "GET",
      });

      // The response from the backend is parsed as JSON
      const searchRes = await response.json();
      console.log(searchRes);
      setError(""); // Clear any existing error messages
    } catch (error) {
      console.log(error);
      setError(error.message);
    }
  };

  // This function handles the submission of the user's prompt when the user hits 'Enter' or 'Submit'
  // It sends a POST request to the provided endpoint with the current prompt in the request body
  const handleSubmitPrompt = async (endpoint) => {
    try {
      setPrompt("");

      // Push the user's message into the messages array
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: prompt, type: "user", sourceDocuments: null },
      ]);

      // A POST request is sent to the backend with the current prompt in the request body
      const response = await fetch(`/api/${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: prompt }),
      });

      // Throw an error if the HTTP status is not OK
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Parse the response from the backend as JSON
      const searchRes = await response.json();

      console.log({ searchRes });

      // Push the response into the messages array
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          text: searchRes.result.text,
          type: "bot",
          sourceDocuments: searchRes.result.sourceDocuments,
        },
      ]);

      setError(""); // Clear any existing error messages
    } catch (error) {
      console.log(error);
      setError(error.message);
    }
  };

  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    console.log("local url: ", URL.createObjectURL(file));
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first.");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const { data } = await axios.post("/api/pdf-upload", formData);
    } catch (err) {
      console.log("failed to upload file", err);
    }
  };

  const resetPineconeIndex = async () => {
    try {
      const { queryResponse } = await axios.post("/api/reset-pinecone-index");
      console.log('cleint deleteIndex resposne', queryResponse)
    } catch (err) {
      console.log("failed to delete index", err);
    }
  }

  // The component returns a two column layout with various child components
  return (
    <>
      <Title emoji="💬" headingText="PDF-GPT" />
      <TwoColumnLayout
        leftChildren={
          <>
            <PageHeader
              heading="Ask Ai about your Documents"

              description="This tool will
            let you ask anything contained in a document. This tool uses
            Embeddings, Pinecone, VectorDBQAChain, and VectorStoreAgents."
            />
            <ButtonContainer>
              {/* <Button
                handleSubmit={()=>{handleSubmit('pdfupload-book')}}
                endpoint="pdfuploadtest"
                buttonText="Upload Test Data ☁️"
                className="Button"
              /> */}
              <Button
                handleSubmit={handleUpload}
                buttonText="Upload File 📚"
                className="Button"
              />
              {/* <Button
                handleSubmit={() => openFilePicker()}
                buttonText="Select a file"
                className="Button"
              /> */}

              <div>
                <h1>Select a file to Upload</h1>
                <input type="file" onChange={handleFileChange} />
              </div>
            </ButtonContainer>
          </>
        }
        rightChildren={
          <>
            <ResultWithSources messages={messages} pngFile="pdf" />
            <PromptBox
              prompt={prompt}
              handlePromptChange={handlePromptChange}
              handleSubmit={() => handleSubmitPrompt("/pdf-query")}
              // handleSubmit={() => handleSubmitQuery("/pdfquery-agent")}
              placeHolderText={"How does XXX work?"}
              error={error}
            />
          </>
        }
      />
    </>
  );
};

export default PDFLoader;
