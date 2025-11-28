import { useState, useEffect } from "react";
import { X, Loader2, FileText, Folder, Image, Download, ChevronRight, Home, ArrowUp } from "lucide-react";
import { FileItem, listFiles, readFile, getDownloadUrl, getZipDownloadUrl } from "@/lib/api";
import { cn } from "@/lib/utils";

interface ArtifactsBrowserProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ArtifactsBrowser({ isOpen, onClose }: ArtifactsBrowserProps) {
  const [currentPath, setCurrentPath] = useState(".");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<FileItem | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadFiles(currentPath);
    }
  }, [isOpen, currentPath]);

  const loadFiles = async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await listFiles(path);
      setFiles(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load files");
    } finally {
      setLoading(false);
    }
  };

  const handleNavigate = (path: string) => {
    setCurrentPath(path);
    setSelectedFile(null);
    setFileContent(null);
  };

  const handleUp = () => {
    if (currentPath === "." || currentPath === "") return;
    const parts = currentPath.split("/");
    parts.pop();
    const newPath = parts.length === 0 || (parts.length === 1 && parts[0] === "") ? "." : parts.join("/");
    handleNavigate(newPath);
  };

  const handleFileClick = async (file: FileItem) => {
    if (file.type === "directory") {
      handleNavigate(file.path);
    } else {
      setSelectedFile(file);
      setLoadingContent(true);
      setFileContent(null);

      // Check if text or image
      const isImage = /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(file.name);

      if (!isImage) {
        try {
          const { content } = await readFile(file.path);
          setFileContent(content);
        } catch (err) {
           // Binary file or error
           setFileContent(null);
        }
      }
      setLoadingContent(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="w-full max-w-5xl h-[80vh] bg-[#1d1d1f] border border-[#333] rounded-xl flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#333] bg-[#1d1d1f]">
          <div className="flex items-center gap-3">
             <h2 className="text-lg font-medium text-white">Project Artifacts</h2>
             <div className="h-4 w-[1px] bg-[#333]" />
             <div className="flex items-center gap-2 text-sm text-[#86868b] overflow-hidden">
                <button onClick={() => handleNavigate(".")} className="hover:text-white transition-colors">
                    <Home className="w-4 h-4" />
                </button>
                {currentPath !== "." && (
                    <>
                         <ChevronRight className="w-3 h-3" />
                         <span className="truncate max-w-[300px]">{currentPath}</span>
                    </>
                )}
             </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-[#333] rounded-full transition-colors">
            <X className="w-5 h-5 text-[#86868b]" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">

            {/* File List */}
            <div className={cn("flex-col border-r border-[#333] bg-[#151516] overflow-y-auto w-full", selectedFile ? "w-1/3 flex" : "w-full flex")}>
                {/* Toolbar */}
                <div className="flex items-center justify-between p-3 border-b border-[#333] sticky top-0 bg-[#151516] z-10">
                    <button
                        onClick={handleUp}
                        disabled={currentPath === "."}
                        className="p-2 text-[#86868b] hover:text-white hover:bg-[#333] rounded-md disabled:opacity-30 disabled:hover:bg-transparent"
                    >
                        <ArrowUp className="w-4 h-4" />
                    </button>
                    <a
                        href={getZipDownloadUrl(currentPath)}
                        className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium bg-[#333] hover:bg-[#444] text-white rounded-md transition-colors"
                        download
                    >
                        <Download className="w-3 h-3" />
                        Download Zip
                    </a>
                </div>

                {loading ? (
                    <div className="flex-1 flex items-center justify-center">
                        <Loader2 className="w-6 h-6 animate-spin text-[#86868b]" />
                    </div>
                ) : error ? (
                    <div className="flex-1 flex items-center justify-center text-red-400 text-sm p-4 text-center">
                        {error}
                    </div>
                ) : files.length === 0 ? (
                    <div className="flex-1 flex items-center justify-center text-[#86868b] text-sm">
                        Empty directory
                    </div>
                ) : (
                    <div className="p-2 space-y-1">
                        {files.map((file) => (
                            <button
                                key={file.path}
                                onClick={() => handleFileClick(file)}
                                className={cn(
                                    "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-left transition-colors group",
                                    selectedFile?.path === file.path ? "bg-[#333] text-white" : "text-[#d1d1d6] hover:bg-[#2c2c2e]"
                                )}
                            >
                                {file.type === "directory" ? (
                                    <Folder className="w-4 h-4 text-blue-400 shrink-0" />
                                ) : /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(file.name) ? (
                                    <Image className="w-4 h-4 text-purple-400 shrink-0" />
                                ) : (
                                    <FileText className="w-4 h-4 text-[#86868b] shrink-0" />
                                )}
                                <span className="truncate flex-1">{file.name}</span>
                                <span className="text-[10px] text-[#555] group-hover:text-[#86868b] whitespace-nowrap">
                                    {file.type === "file" && formatBytes(file.size)}
                                </span>
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Preview Pane */}
            {selectedFile && (
                <div className="flex-1 flex flex-col bg-[#1d1d1f] overflow-hidden animate-in fade-in slide-in-from-right-4 duration-300">
                    <div className="flex items-center justify-between px-6 py-3 border-b border-[#333]">
                        <div className="flex items-center gap-3 overflow-hidden">
                             {/\.(jpg|jpeg|png|gif|webp|svg)$/i.test(selectedFile.name) ? (
                                <Image className="w-4 h-4 text-purple-400 shrink-0" />
                            ) : (
                                <FileText className="w-4 h-4 text-[#86868b] shrink-0" />
                            )}
                            <span className="font-medium text-white truncate">{selectedFile.name}</span>
                        </div>
                        <a
                            href={getDownloadUrl(selectedFile.path)}
                            className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium bg-white text-black hover:bg-[#e5e5e5] rounded-md transition-colors"
                            download
                        >
                            <Download className="w-3 h-3" />
                            Download
                        </a>
                    </div>

                    <div className="flex-1 overflow-auto p-6 relative">
                        {loadingContent ? (
                             <div className="absolute inset-0 flex items-center justify-center">
                                <Loader2 className="w-8 h-8 animate-spin text-[#86868b]" />
                            </div>
                        ) : /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(selectedFile.name) ? (
                            <div className="flex items-center justify-center min-h-full">
                                <img
                                    src={getDownloadUrl(selectedFile.path)}
                                    alt={selectedFile.name}
                                    className="max-w-full max-h-full object-contain rounded-lg shadow-xl"
                                />
                            </div>
                        ) : fileContent !== null ? (
                             <pre className="text-sm font-mono text-[#d1d1d6] whitespace-pre-wrap break-words bg-[#151516] p-4 rounded-lg border border-[#333]">
                                {fileContent}
                            </pre>
                        ) : (
                            <div className="flex flex-col items-center justify-center h-full text-[#86868b] space-y-4">
                                <p>Preview not available for this file type.</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
      </div>
    </div>
  );
}

function formatBytes(bytes: number, decimals = 2) {
  if (!+bytes) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}
