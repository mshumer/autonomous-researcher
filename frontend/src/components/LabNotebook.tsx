import { useState, useEffect, useRef } from "react";
import { Play, StopCircle, Plus } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useExperiment } from "@/lib/useExperiment";
import { AgentNotebook } from "./Notebook/AgentNotebook";
import { ResearchPaper } from "./Notebook/ResearchPaper";
import { cn } from "@/lib/utils";

export function LabNotebook() {
  const { isRunning, agents, orchestrator, startExperiment } = useExperiment();
  const [task, setTask] = useState("");
  const [mode, setMode] = useState<"single" | "orchestrator">("orchestrator");
  const [testMode, setTestMode] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const prevTimelineLengthRef = useRef(0);

  // Auto-scroll effect
  useEffect(() => {
    const currentLength = orchestrator.timeline.length;
    const prevLength = prevTimelineLengthRef.current;

    if (currentLength > prevLength) {
      const lastItem = orchestrator.timeline[currentLength - 1];
      if (lastItem.type === "agents" || lastItem.type === "paper" || currentLength === 1) {
        // Scroll slightly above the new element to keep context
        // We do this by scrolling to the bottom ref, but with 'start' block alignment if possible,
        // or just letting the padding handle it.
        // Actually, let's scroll to the *element itself* if we could, but since we use bottomRef,
        // let's just scroll smoothly to it. 
        // The user said it "goes a little too far", implying it might be scrolling past the top of the new content.
        // Or maybe it scrolls so the bottom is at the bottom of the screen?
        // "scrollIntoView" aligns the element to the top or bottom. 
        
        // Let's try aligning the bottomRef to the 'end' of the view, but give it some breathing room.
        bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      }
    }
    
    prevTimelineLengthRef.current = currentLength;
  }, [orchestrator.timeline]);

  const handleStart = () => {
    if (!task.trim()) return;
    startExperiment(mode, {
      task,
      gpu: "any",
      num_agents: 3,
      max_rounds: 3,
      max_parallel: 2,
      test_mode: testMode,
    });
  };

  return (
    <div className="flex-1 h-screen overflow-hidden flex flex-col bg-black font-sans text-[#f5f5f7] selection:bg-[#333] selection:text-white">
      
      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-hidden relative">

        {/* Sticky Header for Active Research */}
        {orchestrator.timeline.length > 0 && (
            <div className="absolute top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-xl border-b border-white/5 animate-in fade-in slide-in-from-top-4 duration-500">
                <div className="max-w-5xl mx-auto px-8 py-4 flex items-center gap-4">
                    <span className="text-[10px] font-medium text-[#424245] uppercase tracking-widest shrink-0">
                        Objective
                    </span>
                    <p className="text-sm font-light text-[#e5e5e5] truncate">
                        {task}
                    </p>
                </div>
            </div>
        )}
        
        {/* Scrollable Timeline */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
            <div className="max-w-5xl mx-auto py-24 px-8 space-y-32">
                
                {/* Initial Input State (Only visible when timeline is empty) */}
                {orchestrator.timeline.length === 0 && (
                    <div className="min-h-[60vh] flex flex-col justify-center items-center space-y-12 animate-in fade-in duration-1000">
                        <div className="space-y-6 text-center max-w-lg">
                            <h1 className="text-4xl md:text-5xl font-light tracking-tight text-white">
                                Research Objective
                            </h1>
                            <p className="text-lg text-[#86868b] font-light leading-relaxed">
                                Describe your scientific query. The orchestrator will decompose it into hypotheses and launch autonomous agents to investigate.
                            </p>
                        </div>

                        <div className="w-full max-w-xl space-y-8">
                            <div className="relative group">
                                <div className="absolute -inset-1 bg-gradient-to-r from-[#333] to-[#1d1d1f] rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-1000"></div>
                                <textarea
                                    value={task}
                                    onChange={(e) => setTask(e.target.value)}
                                    disabled={isRunning}
                                    placeholder="e.g., Investigate the scaling laws of sparse attention mechanisms..."
                                    className="relative w-full h-32 bg-black border border-[#333] rounded-xl p-6 text-lg font-light text-white placeholder:text-[#333] focus:ring-0 focus:border-[#666] focus:outline-none resize-none leading-relaxed transition-all duration-300"
                                />
                            </div>

                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <button
                                        onClick={() => setTestMode(!testMode)}
                                        className={cn(
                                            "text-[10px] font-medium px-3 py-1.5 rounded-full transition-all duration-300 border border-transparent",
                                            testMode 
                                                ? "bg-white text-black border-white" 
                                                : "bg-[#1d1d1f] text-[#86868b] hover:text-white border-[#333]"
                                        )}
                                    >
                                        {testMode ? "TEST MODE" : "LIVE MODE"}
                                    </button>
                                    <div className="h-4 w-[1px] bg-[#333]" />
                                    <select
                                        value={mode}
                                        onChange={(e) => setMode(e.target.value as "single" | "orchestrator")}
                                        className="bg-transparent text-[#86868b] text-xs font-medium focus:outline-none cursor-pointer hover:text-white transition-colors"
                                    >
                                        <option value="single">Single Agent</option>
                                        <option value="orchestrator">Agent Swarm</option>
                                    </select>
                                </div>

                                <button
                                    onClick={handleStart}
                                    disabled={!task.trim()}
                                    className={cn(
                                        "px-8 py-3 rounded-full text-xs font-medium tracking-widest uppercase transition-all duration-500 flex items-center gap-2",
                                        !task.trim()
                                            ? "bg-[#1d1d1f] text-[#333] cursor-not-allowed"
                                            : "bg-white text-black hover:bg-[#e5e5e5] hover:scale-105"
                                    )}
                                >
                                    <Play className="w-3 h-3 fill-current" />
                                    <span>Start Research</span>
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Timeline Rendering */}
                {orchestrator.timeline.map((item, index) => {
                    if (item.type === "thought") {
                        return (
                            <div key={index} className="w-full animate-in fade-in slide-in-from-bottom-8 duration-700">
                                <div className="pl-6 border-l border-[#333] py-2">
                                    <span className="block text-[10px] font-medium text-[#424245] uppercase tracking-widest mb-3">
                                        Orchestrator
                                    </span>
                                    <div className="prose prose-invert prose-lg md:prose-xl max-w-none prose-p:text-[#d1d1d6] prose-p:font-light prose-p:leading-relaxed prose-strong:text-white prose-headings:text-white prose-code:text-[#d1d1d6] prose-pre:bg-[#1d1d1f] prose-pre:border prose-pre:border-[#333]">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {item.content}
                                        </ReactMarkdown>
                                    </div>
                                </div>
                            </div>
                        );
                    } else if (item.type === "agents") {
                        return (
                            <div key={index} className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700">
                                <div className="flex items-center gap-3">
                                    <div className="h-[1px] w-8 bg-[#333]" />
                                    <span className="text-[10px] font-medium text-[#424245] uppercase tracking-widest">
                                        Sub-Agents Deployed
                                    </span>
                                    <div className="h-[1px] flex-1 bg-[#333]" />
                                </div>
                                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                                    {item.agentIds.map((agentId) => {
                                        const agent = agents[agentId];
                                        if (!agent) return null;
                                        return (
                                            <div key={agentId} className="h-[600px]">
                                                <AgentNotebook agent={agent} />
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        );
                    } else if (item.type === "paper") {
                        return <ResearchPaper key={index} content={item.content} />;
                    }
                    return null;
                })}
                
                {/* Running Indicator at Bottom */}
                {isRunning && orchestrator.timeline.length > 0 && (
                    <div className="flex justify-center py-12">
                        <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-[#1d1d1f] border border-[#333]">
                            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                            <span className="text-[10px] font-medium text-[#86868b] uppercase tracking-widest">
                                Orchestrating
                            </span>
                        </div>
                    </div>
                )}

                <div ref={bottomRef} className="h-10" />
            </div>
        </div>
      </main>

      {/* Minimal Fixed Header (Only visible when running) */}
      {isRunning && (
          <div className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 animate-gradient-x" />
      )}
    </div>
  );
}
