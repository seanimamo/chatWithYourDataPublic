import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PineconeClient } from "@pinecone-database/pinecone";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { CharacterTextSplitter } from "langchain/text_splitter";
import formidable from "formidable";

export const config = {
  api: {
    bodyParser: false,
  },
};

// Example: https://js.langchain.com/docs/modules/indexes/document_loaders/examples/file_loaders/pdf
export default async function handler(req, res) {
  if (req.method === "POST") {
    console.log("Inside the PDF handler");
    const form = formidable();

    form.parse(req, async (err, fields, files) => {
      if (err) {
        return res.status(400).json({ error: "Error parsing the form" });
      }

      if (files.file.length === 0) {
        return res.status(400).json({ error: "No file provided" });
      }

      for (const file of files.file) {
        console.log("processing file", file.originalFilename)
        if (file.mimetype === "application/pdf") {
          console.log("calling PDFLoader with filepath", file.filepath);
          const documents = await loadPdfIntoDocuments(file.filepath);
          await saveDocumentsToVectorStore(documents);
        }
        if (file.mimetype === "text/plain") {
          const documents = await loadTextFileIntoDocuments(file.filepath);
          await saveDocumentsToVectorStore(documents);
        } else {
          console.log("Skipping unsupported file type", file.mimetype);
        }
      }

      // For demonstration, let's assume we've successfully handled the file
      // res.status(200).json({ message: "File uploaded successfully" });
    });

    form.once("end", () => {
      return res.status(200).json({ message: "File uploaded successfully" });
    });
  } else {
    res.status(405).json({ message: "Method not allowed" });
  }
}

async function loadPdfIntoDocuments(filePath) {
  const loader = new PDFLoader(filePath);

  const docs = await loader.load();

  console.log("docs", docs);

  if (docs.length === 0) {
    console.log("no docs found");
    return;
  }
  // Chunk it
  const splitter = new CharacterTextSplitter({
    separator: " ",
    chunkSize: 250, // how big the context is on that page
    chunkOverlap: 10, //how much of each page overlaps onto another
  });

  const splitDocs = await splitter.splitDocuments(docs);
  // Reduce the size of the metadata
  const reduceDocs = splitDocs.map((doc) => {
    const reducedMetadata = { ...doc.metadata };
    delete reducedMetadata.pdf;
    return new Document({
      pageContent: doc.pageContent,
      metadata: reducedMetadata,
    });
  });

  return reduceDocs;
}

async function loadTextFileIntoDocuments(filePath) {
  const loader = new TextLoader(filePath);

  const docs = await loader.load();

  console.log("docs", docs);

  if (docs.length === 0) {
    console.log("no docs found");
    return;
  }
  // Chunk it
  const splitter = new CharacterTextSplitter({
    separator: " ",
    chunkSize: 250, // how big the context is on that page
    chunkOverlap: 10, //how much of each page overlaps onto another
  });

  const splitDocs = await splitter.splitDocuments(docs);
  // Reduce the size of the metadata
  const reduceDocs = splitDocs.map((doc) => {
    const reducedMetadata = { ...doc.metadata };
    delete reducedMetadata.pdf;
    return new Document({
      pageContent: doc.pageContent,
      metadata: reducedMetadata,
    });
  });

  return reduceDocs;
}

async function saveDocumentsToVectorStore(documents) {
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
  });

  const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

  // upload documents to Pinecone
  // OpenAIEmbeddings is automatically using our OPENAI_API_KEY to turn the document into vector embeddings
  await PineconeStore.fromDocuments(documents, new OpenAIEmbeddings(), {
    pineconeIndex,
  });

  console.log("Successfully uploaded to database");
}

async function loadAndSplitAndSavePDF(filePath) {
  const loader = new PDFLoader(filePath);

  const docs = await loader.load();

  console.log("docs", docs);

  if (docs.length === 0) {
    console.log("no docs found");
    return;
  }
  // Chunk it
  const splitter = new CharacterTextSplitter({
    separator: " ",
    chunkSize: 250, // how big the context is on that page
    chunkOverlap: 10, //how much of each page overlaps onto another
  });

  const splitDocs = await splitter.splitDocuments(docs);
  // Reduce the size of the metadata
  const reduceDocs = splitDocs.map((doc) => {
    const reducedMetadata = { ...doc.metadata };
    delete reducedMetadata.pdf;
    return new Document({
      pageContent: doc.pageContent,
      metadata: reducedMetadata,
    });
  });

  /** STEP TWO: UPLOAD TO DATABASE */
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
  });

  const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

  // upload documents to Pinecone
  // OpenAIEmbeddings is automatically using our OPENAI_API_KEY to turn the document into vector embeddings
  await PineconeStore.fromDocuments(reduceDocs, new OpenAIEmbeddings(), {
    pineconeIndex,
  });

  console.log("Successfully uploaded to database");
}
