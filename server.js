import express from "express"
import dotenv from "dotenv";
import path from "path";


import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {z} from "zod";

dotenv.config();
const port = 3000;
const app = express();
app.use(express.json());

const __dirname = path.resolve();

const model = new ChatGoogleGenerativeAI({
    model: "models/gemini-2.5-flash", // Free-tier model - use pro
    maxOutputTokens: 2048,
    temperature: 0.7,
    apiKey: process.env.GOOGLE_GEMINI_API_KEY,
});
//Tool For Restaurant Ai
const getMenuTool = new DynamicStructuredTool({
    name: "getMenu",
    description: "Returns the final answer for today's menu. Use this tool if someone asks about today's breakfast, lunch, or dinner.",
    schema: z.object({
        category: z.string().describe("Type of food..."),
    }),
    func: async ({ category }) => {
    const menus = {
        breakfast: "Aloo Paratha, Poha, Masala Chai",
        lunch: "Paneer Butter Masala, Dal Fry, Jeera Rice, Roti",
        dinner: "Veg Biryani, Raita, Salad, Gulab Jamun"
    };
    return menus[category.toLowerCase()] || "No menu found for that";
},
    // ...the input that the tool needs.
})
const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that uses tools when needed."],
    ["human", "{input}"],
    ["ai", "{agent_scratchpad}"]
]);

const agent = await createToolCallingAgent({
  llm: model,
  tools: [getMenuTool],
  prompt
})

const executor = await AgentExecutor.fromAgentAndTools({
  agent,
  tools: [getMenuTool],
  verbose: true,
  maxIterations: 1,
  returnIntermediateSteps: true
})
app.get("/", (req,res)=>{
    return res.sendFile(path.join(__dirname,'public','index.html'))
    
});

app.post('/api/chat',async (req,res) => {
    const userInput = req.body.input;
    console.log("userInput : ", userInput);

    try {
        const response = await executor.invoke({input: userInput});
        console.log("Agent full Response : ", response);
        const data = response.intermediateSteps[0].observation;

        if (response.output && response.output != "Agent stopped due to max iterations."){
            return res.json({output: response.output});
        } else if (data != null){
            return res.json({output: data});
        }
        res.status(500).json({output: "Agent couldn't find a valid answer."})
    }
    catch (err){
        console.log("Error during agent execution",err);
        res.status(500).json({output: "sorry, something went wrong. please try again."})
    }
})


app.listen(3000, () => {
  console.log(`Server is running on port http://localhost:${port}`);
});
