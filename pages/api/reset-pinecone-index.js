import { PineconeClient } from "@pinecone-database/pinecone";
import { VectorDBQAChain } from "langchain/chains";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";

// Example: https://js.langchain.com/docs/modules/indexes/document_loaders/examples/file_loaders/pdf
export default async function handler(req, res) {
  try {
    if (req.method !== "POST") {
      throw new Error("Method not allowed");
    }

    // Initialize Pinecone
    const client = new PineconeClient();
    await client.init({
      apiKey: process.env.PINECONE_API_KEY,
      environment: process.env.PINECONE_ENVIRONMENT,
    });

    const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

    const allVectorsRequest = {
      topK: 10000,
      vector: new Array(1536).fill(0),
      includeMetadata: true,
      includeValues: false,
    };

    const queryResponse = await pineconeIndex.query({
      queryRequest: allVectorsRequest,
    });


    console.log('found query matches', queryResponse)

    // TODO: Delete the vectors

    // Search!

    return res.status(200).json({ result: queryResponse });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: error.message });
  }
}
