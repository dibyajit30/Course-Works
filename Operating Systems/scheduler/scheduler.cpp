#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <list>
#include <iomanip>
#include <ctype.h>
#include <stdlib.h>

using namespace std;

ifstream input;
char * delimiter = " ";
int randomCount;
bool v_trace = false;
bool e_trace = false;
bool t_trace = false;
int quantum = 10000;
int maxPrio = 4;
int randomOfset = 0;
enum scheduler_categories {FCFS = 'F', LCFS = 'L', SRTF = 'S', RR = 'R', PRIO = 'P', PREPRIO = 'E'};
enum state {created, ready, running, blocked, done};

struct process
{
    int pid,at,tc,cb,io,static_prio,dyn_prio,ft,tt,current_cb,current_io,remaining_CPUTime,remaining_quant;
    int it = 0;
    int cw = 0;
    state current_state = created;
};

vector<int> randomValues;
vector<process> createdProcesses;

int myrandom(int burst){
    if(randomOfset >= randomCount){
        randomOfset = 0;
    }
    return 1 + (randomValues[randomOfset++] % burst); 
}

class scheduler{
    int finishTime;
    double Avg_turnAroundTime, Avg_waitTime, throughput_percent; 
    double CPU_util = 0.0;
    double IO_util = 0.0;
    
    protected:
    int timestamp=0;
    list<process*> readyQueue;
    vector<process*> blockedQueue;
    process* runningProcess = NULL;
    
    int getMinQuant_nextEvent(){
        int minQuant = 10000;
        // next arrival time
        for(int i=0; i<createdProcesses.size(); i++){
            if((createdProcesses[i].current_state == created) && (createdProcesses[i].at-timestamp < minQuant)){
                minQuant = createdProcesses[i].at-timestamp;
            }
        }
        // next IO burst 
        for(int i=0; i<blockedQueue.size(); i++){
            if((*blockedQueue[i]).current_io < minQuant){
                minQuant = (*blockedQueue[i]).current_io;
            }
        }
        // next Premption
        if((runningProcess != NULL) && ((*runningProcess).remaining_quant < minQuant)){
            minQuant = (*runningProcess).remaining_quant;
        }
        // next CPU burst
        if((runningProcess != NULL) && ((*runningProcess).current_cb < minQuant)){
            minQuant = (*runningProcess).current_cb;
        }
        return minQuant;
    }

    void applyQuantTime(int quantTime){
        if(quantTime == 10000){
            quantTime = 0;
        }
        timestamp += quantTime;
        for(int i=0; i<createdProcesses.size(); i++){
            switch (createdProcesses[i].current_state)
            {
            case ready:
                createdProcesses[i].cw += quantTime;
                break;
            
            case running:
                createdProcesses[i].remaining_CPUTime -= quantTime;
                createdProcesses[i].remaining_quant -= quantTime;
                createdProcesses[i].current_cb -= quantTime;
                break;
            
            case blocked:
                createdProcesses[i].current_io -= quantTime;
                createdProcesses[i].it += quantTime;
                break;
            
            default:
                break;
            }
        }
        if(runningProcess != NULL){
            CPU_util += quantTime;
        }
        if(!blockedQueue.empty()){
            IO_util += quantTime;
        }
    }    

    void display_summary(scheduler_categories scheduler_type){
        switch (scheduler_type)
        {
        case FCFS:
            cout<<"FCFS"<<endl;
            break;
        case LCFS:
            cout<<"LCFS"<<endl;
            break;
        case SRTF:
            cout<<"SRTF"<<endl;
            break;
        case RR:
            cout<<"RR "<<quantum<<endl;
            break;
        case PRIO:
            cout<<"PRIO "<<quantum<<endl;
            break;
        case PREPRIO:
            cout<<"PREPRIO "<<quantum<<endl;
            break;
        default:
            break;
        }
        for(int i=0; i<createdProcesses.size(); i++){
            cout<<setfill('0')<<setw(4)<<createdProcesses[i].pid<<":";
            cout<<setfill(' ')<<setw(5)<<createdProcesses[i].at;
            cout<<setfill(' ')<<setw(5)<<createdProcesses[i].tc;
            cout<<setfill(' ')<<setw(5)<<createdProcesses[i].cb;
            cout<<setfill(' ')<<setw(5)<<createdProcesses[i].io;
            cout<<" "<<createdProcesses[i].static_prio;
            cout<<" |";
            cout<<setfill(' ')<<setw(6)<<createdProcesses[i].ft;
            cout<<setfill(' ')<<setw(6)<<createdProcesses[i].tt;
            cout<<setfill(' ')<<setw(6)<<createdProcesses[i].it;
            cout<<setfill(' ')<<setw(6)<<createdProcesses[i].cw;
            cout<<endl;
        }
        calculate_summary();
        printf("SUM: %d %.2lf %.2lf %.2lf %.2lf %.3lf", timestamp, (CPU_util/timestamp)*100, (IO_util/timestamp)*100, Avg_turnAroundTime, Avg_waitTime, throughput_percent);
    }

    void calculate_summary(){
        int size = createdProcesses.size();
        double turnaroundTime = 0.0;
        double cpuWaitTime = 0.0;
        for(int i=0; i<size; i++){
            turnaroundTime += createdProcesses[i].tt;
            cpuWaitTime += createdProcesses[i].cw;
        }
        Avg_turnAroundTime = (double)turnaroundTime/size;
        Avg_waitTime = (double)cpuWaitTime/size;
        throughput_percent = (double)(size*100)/timestamp;
    }

};

class fcfs: protected virtual scheduler{
    protected:
    void addProcessToReadyQueue(process * currentProcess){
        readyQueue.push_back(currentProcess);
    }
};

class lcfs: protected virtual scheduler{
    protected:
    void addProcessToReadyQueue(process * currentProcess){
        readyQueue.push_front(currentProcess);
    }
};

class srtf: protected virtual scheduler{

    protected:
    void addProcessToReadyQueue(process * currentProcess){
        int min_remTime = 10000;
        if(readyQueue.empty()){
            readyQueue.push_back(currentProcess);
        }
        else{
            list<process*>::iterator i;
            bool added = false;
            for(i=readyQueue.begin(); i!=readyQueue.end(); ++i){
                if((*(*i)).remaining_CPUTime > (*currentProcess).remaining_CPUTime){
                    readyQueue.insert(i, currentProcess);
                    added = true;
                    break;
                }
            }
            if(!added){
                readyQueue.push_back(currentProcess);
            }
        } 
    }
};

class round_robin: protected virtual scheduler{
    protected:
    void addProcessToReadyQueue(process * currentProcess){
        readyQueue.push_back(currentProcess);
    }
};

class prio: protected virtual scheduler{
    protected:
    list<process*> expiredQueue;
    void addProcessOnPrio(list<process*>* processList, process * currentProcess){
        if((*processList).empty()){
            (*processList).push_back(currentProcess);
        }
        else{
            list<process*>::iterator i;
            bool added = false;
            for(i=(*processList).begin(); i!=(*processList).end(); ++i){
                if((*(*i)).dyn_prio < (*currentProcess).dyn_prio){
                    (*processList).insert(i, currentProcess);
                    added = true;
                    break;
                }
            }
            if(!added){
                (*processList).push_back(currentProcess);
            }
        }
    }
    void addProcessToReadyQueue(process * currentProcess){
        if((*currentProcess).current_cb != 0){
            (*currentProcess).dyn_prio -= 1;
        }
        if((*currentProcess).dyn_prio < 0){
            (*currentProcess).dyn_prio = (*currentProcess).static_prio - 1;
            addProcessOnPrio(&expiredQueue, currentProcess);
        }
        else{
            if((*currentProcess).current_cb == 0){
                (*currentProcess).dyn_prio = (*currentProcess).static_prio - 1;
            }
            addProcessOnPrio(&readyQueue, currentProcess);
        }
        
    }
};

class simulate: fcfs, lcfs, srtf, round_robin, prio{
    scheduler_categories scheduler_type;
    int min_quant;
    int prevQuant = 0;
    public:
    simulate(scheduler_categories scheduler_type){
        this->scheduler_type = scheduler_type;
    }

    bool getEvents(){
        bool isEvent = false;
        for(int i=0; i<createdProcesses.size(); i++){
            if((createdProcesses[i].current_state == created) && (createdProcesses[i].at == timestamp)){
                // add process i from created to ready queue
                created_ready(&createdProcesses[i]);
                isEvent = true;
            }
        }
        for(int i=0; i<blockedQueue.size(); i++){
            if((*blockedQueue[i]).current_io == 0){
                // add process i from blocked to ready queue
                blocked_ready(blockedQueue[i]);
                i -= 1;
                isEvent = true;
            }
        }
        if((runningProcess != NULL) && ((*runningProcess).current_cb == 0)){
            if((*runningProcess).remaining_CPUTime == 0){
                // done
                running_done(runningProcess);
                isEvent = true;
            }
            else{
                // add process i to blocked queue
                running_blocked(runningProcess);
                isEvent = true;
            }
        }
        if((runningProcess != NULL) && ((*runningProcess).remaining_quant == 0)){
            // add process i from running to ready queue
            running_ready(runningProcess);
            isEvent = true;
        }
        if((runningProcess == NULL) && (readyQueue.size() || expiredQueue.size())){
            // get next process from ready queue
            ready_running();
            isEvent = true;
        }
        prevQuant = min_quant;
        return isEvent;
    }

    void created_ready(process * currentProcess){
        (*currentProcess).current_state = ready;
        (*currentProcess).current_cb = 0;
        if(v_trace){
            cout<<timestamp<<" "<<(*currentProcess).pid<<" "<<min_quant<<": CREATED -> READY"<<endl;
        }
        addProcessToReadyQueue(currentProcess);
    }

    void blocked_ready(process * currentProcess){
        (*currentProcess).current_state = ready;
        addProcessToReadyQueue(currentProcess);
        vector<process*>::iterator i;
        for(i=blockedQueue.begin(); i!=blockedQueue.end(); ++i){
            if((*i) == currentProcess){
                blockedQueue.erase(i);
                break;
            }
        }
        if(v_trace){
            cout<<timestamp<<" "<<(*currentProcess).pid<<" "<<min_quant<<": BLOCK -> READY"<<endl;
        }
    }

    void running_ready(process * currentProcess){
        (*currentProcess).current_state = ready;
        if(v_trace){
            cout<<timestamp<<" "<<(*currentProcess).pid<<" "<<min_quant<<": RUNNG -> READY";
            cout<<"  cb="<<(*currentProcess).current_cb<<" rem="<<(*currentProcess).remaining_CPUTime<<" prio="<<(*currentProcess).dyn_prio<<endl;
        }
        runningProcess = NULL;
        addProcessToReadyQueue(currentProcess);
    }

    void addProcessToReadyQueue(process * currentProcess){
        switch (scheduler_type)
        {
        case FCFS:
            fcfs::addProcessToReadyQueue(currentProcess);
            break;
        case LCFS:
            lcfs::addProcessToReadyQueue(currentProcess);
            break;
        case SRTF:
            srtf::addProcessToReadyQueue(currentProcess);
            break;
        case RR:
            round_robin::addProcessToReadyQueue(currentProcess);
            break;
        case PRIO:
            prio::addProcessToReadyQueue(currentProcess);
            break;
        case PREPRIO:
            prio::addProcessToReadyQueue(currentProcess);
            if(runningProcess != NULL){
                if(v_trace){
                    cout<<"---> PRIO preemption "<<(*runningProcess).pid<<" by "<<(*currentProcess).pid<<" ? ";
                    if((*readyQueue.front()).dyn_prio > (*runningProcess).dyn_prio){
                        cout<<1<<" TS="<<timestamp+1<<" now="<<timestamp;
                        if((*runningProcess).current_cb != 0){
                            cout<<") --> YES"<<endl;
                        }
                        else{
                            cout<<") --> NO"<<endl;
                        }
                    }
                    else{
                        cout<<0<<" TS="<<timestamp+1<<" now="<<timestamp<<") --> NO"<<endl;
                    }
                }
                if(((*readyQueue.front()).dyn_prio > (*runningProcess).dyn_prio) && ((*runningProcess).current_cb != 0)){
                    running_ready(runningProcess);
                }
            }
            break;
        default:
            break;
        }
    }

    void running_done(process * currentProcess){
        (*currentProcess).current_state = done;
        (*currentProcess).ft = timestamp;
        (*currentProcess).tt = timestamp - (*currentProcess).at;
        if(v_trace){
            cout<<timestamp<<" "<<(*currentProcess).pid<<" "<<min_quant<<": Done"<<endl;
        }
        runningProcess = NULL;
    }

    void running_blocked(process * currentProcess){
        (*currentProcess).current_state = blocked;
        (*currentProcess).current_io = myrandom((*currentProcess).io);
        blockedQueue.push_back(currentProcess);
        if(v_trace){
            cout<<timestamp<<" "<<(*currentProcess).pid<<" "<<min_quant<<": RUNNG -> BLOCK";
            cout<<"  ib="<<(*currentProcess).current_io<<" rem="<<(*currentProcess).remaining_CPUTime<<endl;
        }
        runningProcess = NULL;
    }

    void ready_running(){
        if(readyQueue.empty()){
            readyQueue = expiredQueue;
            expiredQueue.clear();
        }
        process * currentProcess = readyQueue.front();
        readyQueue.pop_front();
        (*currentProcess).current_state = running;
        if(((*currentProcess).current_cb == 0) ||((*currentProcess).current_cb == (*currentProcess).cb)){
            (*currentProcess).current_cb = myrandom((*currentProcess).cb);
        }
        if((*currentProcess).remaining_CPUTime < (*currentProcess).current_cb){
            (*currentProcess).current_cb = (*currentProcess).remaining_CPUTime;
        }
        (*currentProcess).remaining_quant = quantum;
        runningProcess = currentProcess;
        if(v_trace){
            cout<<timestamp<<" "<<(*currentProcess).pid<<" "<<min_quant<<": READY -> RUNNG";
            cout<<" cb="<<(*currentProcess).current_cb<<" rem="<<(*currentProcess).remaining_CPUTime<<" prio="<<(*currentProcess).dyn_prio<<endl;
        }
    }

    void run_simulator(){
        min_quant = getMinQuant_nextEvent();
        applyQuantTime(min_quant);
        while(getEvents()){
            min_quant = getMinQuant_nextEvent();
            applyQuantTime(min_quant);
        }
        display_summary(scheduler_type);
    }
};

void addCreatedProcess(int at, int tc, int cb, int io){
    struct process newProcess;
    newProcess.at = at;
    newProcess.cb = cb;
    newProcess.io = io;
    newProcess.tc = tc;
    newProcess.pid = createdProcesses.size();
    newProcess.static_prio = myrandom(maxPrio);
    newProcess.dyn_prio = newProcess.static_prio -1;
    newProcess.remaining_CPUTime = tc;
    newProcess.remaining_quant = quantum;
    createdProcesses.push_back(newProcess);
}

void parseProcess(char * line){
    int at, tc, cb, io;
    at = atoi(strtok(line, delimiter));
    tc = atoi(strtok(NULL, delimiter));
    cb = atoi(strtok(NULL, delimiter));
    io = atoi(strtok(NULL, delimiter));
    addCreatedProcess(at, tc, cb, io);
}

void readProcesses(char * fileName){
    input.open(fileName);
    string newLine;
    char * line = "";
    while (getline(input, newLine)){
        line = (char*) malloc(sizeof(char) * newLine.length() + 1);
        strcpy(line, newLine.c_str());
        parseProcess(line);
    }
        input.close(); 
}

void setRandomValues(char * file){
    input.open(file);
    string newLine;
    getline(input, newLine);
    char * line = "";
    line = (char*) malloc(sizeof(char) * newLine.length() + 1);
    strcpy(line, newLine.c_str());
    randomCount = atoi(strtok(line, delimiter));
    for(int i=0; i<randomCount; i++){
        getline(input, newLine);
        line = (char*) malloc(sizeof(char) * newLine.length() + 1);
        strcpy(line, newLine.c_str());
        randomValues.push_back(atoi(strtok(line, delimiter)));
    }
    input.close();
}

scheduler_categories getSchedulerType(char * schToken){
    scheduler_categories type;
    char schAlgorithm = schToken[2];
    schToken[0] = '0';
    schToken[1] = '0';
    schToken[2] = '0';
    switch (schAlgorithm)
    {
    case FCFS:
        type = FCFS;
        break;
    case LCFS:
        type = LCFS;
        break;
    case SRTF:
        type = SRTF;
        break;
    case RR:
        type = RR;
        sscanf(schToken, "%d", &quantum);
        break;
    case PRIO:
        type = PRIO;
        sscanf(schToken, "%d:%d", &quantum, &maxPrio);
        break;
    case PREPRIO:
        type = PREPRIO;
        sscanf(schToken, "%d:%d", &quantum, &maxPrio);
        break;
    default:
        break;
    }
    return type;
}

void setTraces(int args, char ** input){
    for(int i=1; i<args; i++){
        if(input[i][1] == 'v'){
            v_trace = true;
        }
        else if(input[i][1] == 't'){
            t_trace = true;
        }
        else{
            e_trace = true;
        }
    }
}

int main(int argc, char ** inputParam){
    char * processesFile = inputParam[argc - 2];
    char * randomValueFile = inputParam[argc - 1];
    setTraces(argc-3, inputParam);
    scheduler_categories scheduler_type = getSchedulerType(inputParam[argc - 3]);
    setRandomValues(randomValueFile);
    readProcesses(processesFile);
    simulate scheduler_simulate(scheduler_type);
    scheduler_simulate.run_simulator();
    return 0;
}