"use client";

import { useState } from "react";

export default function Home() {
  const [messages, setMessages] = useState<
    { role: "user" | "agent"; content: string }[]
  >([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      /*const res = await fetch("http://127.0.0.1:10000/query", { */
      const res = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: input }),
      });

      if (!res.ok) {
        throw new Error("Failed to fetch data from Backend");
      }

      const data = await res.json();

      console.log(data);

      setMessages((prev) => [
        ...prev,
        { role: "agent", content: data.answer },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "agent", content: "Error contacting backend." },
      ]);
    }

    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-gray-100 p-10">
      <h1 className="text-3xl font-bold mb-6">
        Decision Intelligence Agent
      </h1>

      <div className="bg-white rounded-xl shadow p-6 max-w-3xl mx-auto">
        <div className="space-y-4 mb-6">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`p-4 rounded-lg whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-blue-100 text-right"
                  : "bg-gray-100 text-left"
              }`}
            >
              {msg.content}
            </div>
          ))}

          {loading && (
            <div className="text-gray-500 italic">
              Analyzing item signals...
            </div>
          )}
        </div>

        <div className="flex gap-2">
          <input
            className="flex-1 p-3 border rounded-lg"
            placeholder="Ask about ITEM_100..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button
            onClick={handleSubmit}
            className="px-5 py-2 bg-black text-white rounded-lg"
          >
            Send
          </button>
        </div>
      </div>
    </main>
  );
}