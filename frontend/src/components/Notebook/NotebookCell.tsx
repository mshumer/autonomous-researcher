import { Brain, Terminal, ArrowRight, ChevronRight } from "lucide-react";
import { ExperimentStep } from "@/lib/useExperiment";
import { cn } from "@/lib/utils";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface NotebookCellProps {
    step: ExperimentStep;
}

export function NotebookCell({ step }: NotebookCellProps) {
    const { type, content } = step;

    if (type === "thought") {
        return (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
                <div className="flex items-center gap-2 mb-2">
                    <div className="w-1 h-1 rounded-full bg-[#424245]" />
                    <span className="text-[10px] font-medium text-[#424245] uppercase tracking-widest">Thinking</span>
                </div>
                <div className="pl-3 border-l border-[#1d1d1f] ml-0.5">
                    <div className="prose prose-invert prose-sm max-w-none prose-p:text-[#86868b] prose-p:text-xs prose-p:leading-relaxed prose-p:font-light prose-headings:text-[#d1d1d6] prose-strong:text-[#d1d1d6] prose-code:text-[#d1d1d6] prose-pre:bg-[#1d1d1f] prose-pre:border prose-pre:border-[#333] [&>*:first-child]:mt-0">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {content}
                        </ReactMarkdown>
                    </div>
                </div>
            </div>
        );
    }

    if (type === "code") {
        return (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
                <div className="flex items-center gap-2 mb-2">
                    <Terminal className="w-3 h-3 text-[#424245]" />
                    <span className="text-[10px] font-medium text-[#424245] uppercase tracking-widest">Command</span>
                </div>
                <div className="bg-[#0a0a0a] border border-[#1d1d1f] rounded-sm overflow-hidden">
                    <pre className="p-3 text-[10px] font-mono text-[#d1d1d6] overflow-x-auto custom-scrollbar">
                        {content}
                    </pre>
                </div>
            </div>
        );
    }

    if (type === "result") {
        // Handle carriage returns (\r) for progress bars (like tqdm).
        // We want to simulate the terminal behavior where \r moves the cursor 
        // to the start of the line, allowing subsequent text to overwrite.
        // We split by \r and take the last segment for the current line context if it's a pure overwrite,
        // but \r can be mixed with \n.
        // A simple approximation is: split by \n, and for each line, process \r.
        
        const processCarriageReturns = (text: string) => {
            const lines = text.split('\n');
            const processedLines = lines.map(line => {
                // If line has \r, usually we just want the text AFTER the last \r
                // unless that \r is followed by nothing? 
                // Standard terminal: "Loading... 10%\rLoading... 20%" -> "Loading... 20%"
                // "Item 1\rItem 2" -> "Item 2"
                
                if (line.includes('\r')) {
                    const parts = line.split('\r');
                    return parts[parts.length - 1];
                }
                return line;
            });
            return processedLines.join('\n');
        };

        return (
            <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
                <div className="flex items-center gap-2 mb-2">
                    <ArrowRight className="w-3 h-3 text-[#424245]" />
                    <span className="text-[10px] font-medium text-[#424245] uppercase tracking-widest">Output</span>
                </div>
                <div className="pl-3 border-l border-[#1d1d1f] ml-0.5">
                    <pre className="text-[10px] font-mono text-[#86868b] overflow-x-auto custom-scrollbar whitespace-pre-wrap">
                        {processCarriageReturns(content)}
                    </pre>
                </div>
            </div>
        );
    }

    return null;
}
