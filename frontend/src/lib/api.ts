// In production (Railway), the API is served from the same origin
// In development, we use localhost:8000
export const API_BASE_URL = import.meta.env.DEV ? "http://localhost:8000" : "";

// LocalStorage key for user credentials
const CREDENTIALS_STORAGE_KEY = "ai_researcher_credentials";

export interface UserCredentials {
    google_api_key?: string;
    anthropic_api_key?: string;
    modal_token_id?: string;
    modal_token_secret?: string;
}

export interface SingleExperimentRequest {
    task: string;
    gpu?: string;
    model?: string;
    test_mode?: boolean;
    credentials?: UserCredentials;
}

export interface OrchestratorExperimentRequest {
    task: string;
    gpu?: string;
    model?: string;
    num_agents: number;
    max_rounds: number;
    max_parallel: number;
    test_mode?: boolean;
    credentials?: UserCredentials;
}

export type ExperimentRequest =
    | SingleExperimentRequest
    | OrchestratorExperimentRequest;

export interface LogEvent {
    type: "line" | "summary";
    stream?: "stdout" | "stderr";
    timestamp: string;
    raw?: string;
    plain?: string;
    exit_code?: number;
    duration_seconds?: number;
}

export interface ChartSeries {
    name: string;
    values: number[];
}

export interface ChartSpec {
    title?: string;
    type: "line" | "bar";
    labels: string[];
    series: ChartSeries[];
}

export interface AgentSummaryRequest {
    agent_id: string;
    history: { type: "thought" | "code" | "result" | "text"; content: string }[];
}

export interface AgentSummaryResponse {
    summary: string;
    chart?: ChartSpec | null;
}

export interface CredentialStatus {
    hasGoogleApiKey: boolean;
    hasAnthropicApiKey: boolean;
    hasModalToken: boolean;
}

export interface CredentialUpdatePayload {
    googleApiKey?: string;
    anthropicApiKey?: string;
    modalTokenId?: string;
    modalTokenSecret?: string;
}

// ---------------------------------------------------------------------------
// Local credential storage (browser localStorage)
// ---------------------------------------------------------------------------

export function getStoredCredentials(): UserCredentials {
    try {
        const stored = localStorage.getItem(CREDENTIALS_STORAGE_KEY);
        if (stored) {
            return JSON.parse(stored);
        }
    } catch (e) {
        console.warn("Failed to read stored credentials:", e);
    }
    return {};
}

export function storeCredentials(creds: CredentialUpdatePayload): void {
    const toStore: UserCredentials = {
        google_api_key: creds.googleApiKey,
        anthropic_api_key: creds.anthropicApiKey,
        modal_token_id: creds.modalTokenId,
        modal_token_secret: creds.modalTokenSecret,
    };
    // Only store non-empty values
    const filtered = Object.fromEntries(
        Object.entries(toStore).filter(([, v]) => v && v.trim())
    );
    localStorage.setItem(CREDENTIALS_STORAGE_KEY, JSON.stringify(filtered));
}

export function getLocalCredentialStatus(): CredentialStatus {
    const creds = getStoredCredentials();
    return {
        hasGoogleApiKey: Boolean(creds.google_api_key?.trim()),
        hasAnthropicApiKey: Boolean(creds.anthropic_api_key?.trim()),
        hasModalToken: Boolean(creds.modal_token_id?.trim() && creds.modal_token_secret?.trim()),
    };
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function streamExperiment(
    endpoint: "/api/experiments/single/stream" | "/api/experiments/orchestrator/stream",
    payload: ExperimentRequest,
    onData: (data: LogEvent) => void,
    onError: (error: Error) => void,
    onComplete: () => void
) {
    try {
        // Attach stored credentials to the request
        const credentials = getStoredCredentials();
        const payloadWithCreds = { ...payload, credentials };

        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payloadWithCreds),
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        if (!response.body) {
            throw new Error("No response body");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const event = JSON.parse(line);
                    onData(event);
                } catch (e) {
                    console.warn("Failed to parse JSON line:", line, e);
                }
            }
        }

        onComplete();
    } catch (error) {
        onError(error instanceof Error ? error : new Error(String(error)));
    }
}

export async function summarizeAgent(payload: AgentSummaryRequest): Promise<AgentSummaryResponse> {
    const response = await fetch(`${API_BASE_URL}/api/agents/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error(`Summarizer error: ${response.status} ${response.statusText}`);
    }

    const data = (await response.json()) as AgentSummaryResponse;
    return data;
}

export async function fetchCredentialStatus(): Promise<CredentialStatus> {
    // In multi-user mode, we check localStorage instead of server
    // Server credentials are only used as fallback
    const localStatus = getLocalCredentialStatus();

    // Also check server for fallback (e.g., if server has env vars set)
    try {
        const response = await fetch(`${API_BASE_URL}/api/credentials/status`);
        if (response.ok) {
            const data = await response.json();
            return {
                hasGoogleApiKey: localStatus.hasGoogleApiKey || Boolean(data.has_google_api_key),
                hasAnthropicApiKey: localStatus.hasAnthropicApiKey || Boolean(data.has_anthropic_api_key),
                hasModalToken: localStatus.hasModalToken || Boolean(data.has_modal_token),
            };
        }
    } catch {
        // Server check failed, just use local status
    }

    return localStatus;
}

export async function saveCredentials(payload: CredentialUpdatePayload): Promise<CredentialStatus> {
    // Store credentials locally in the browser
    storeCredentials(payload);

    // Return the new status based on what we just stored + what was already there
    return fetchCredentialStatus();
}

// ---------------------------------------------------------------------------
// File API
// ---------------------------------------------------------------------------

export interface FileItem {
    name: string;
    path: string;
    type: "directory" | "file";
    size: number;
    modified: string;
}

export async function listFiles(path: string = "."): Promise<FileItem[]> {
    const response = await fetch(`${API_BASE_URL}/api/files/list?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
        throw new Error(`Failed to list files: ${response.statusText}`);
    }
    return response.json();
}

export async function readFile(path: string): Promise<{ content: string }> {
    const response = await fetch(`${API_BASE_URL}/api/files/read?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `Failed to read file: ${response.statusText}`);
    }
    return response.json();
}

export function getDownloadUrl(path: string): string {
    return `${API_BASE_URL}/api/files/download?path=${encodeURIComponent(path)}`;
}

export function getZipDownloadUrl(path: string): string {
    return `${API_BASE_URL}/api/files/download_zip?path=${encodeURIComponent(path)}`;
}
