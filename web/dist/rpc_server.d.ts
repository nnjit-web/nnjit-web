declare enum RPCServerState {
    InitHeader = 0,
    InitHeaderKey = 1,
    InitServer = 2,
    WaitForCallback = 3,
    ReceivePacketHeader = 4,
    ReceivePacketBody = 5
}
/**
 * A websocket based RPC
 */
export declare class RPCServer {
    url: string;
    key: string;
    socket: WebSocket;
    state: RPCServerState;
    logger: (msg: string) => void;
    getImports: () => Record<string, unknown>;
    private pendingSend;
    private name;
    private inst?;
    private serverRecvData?;
    private currPacketHeader?;
    private currPacketLength;
    private remoteKeyLength;
    private pendingBytes;
    private buffredBytes;
    private messageQueue;
    constructor(url: string, key: string, getImports: () => Record<string, unknown>, logger?: (msg: string) => void);
    private onClose;
    private onOpen;
    /** Handler for raw message. */
    private onMessage;
    /** Process ready events. */
    private processEvents;
    /** State machine to handle each request */
    private onDataReady;
    private onPacketReady;
    /** Event handler during server initialization. */
    private onInitServer;
    private log;
    private handleInitHeader;
    private handleInitHeaderKey;
    private checkLittleEndian;
    private requestBytes;
    private readFromBuffer;
}
export {};
