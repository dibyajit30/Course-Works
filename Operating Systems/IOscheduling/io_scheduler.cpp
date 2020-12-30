#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <list>
#include <iomanip>
#include <ctype.h>
#include <stdlib.h>
#include <limits.h>

using namespace std;
ifstream input;
char * delimiter = " ";
char * line = "";
bool v_trace = false;
bool q_trace = false;
bool f_trace = false;

struct IOrequest{
    int arrivalTime, track, startTime, endTime, turnaroundTime;
};

vector<IOrequest> io_requests;

void lineReader(){
    string newLine;
    if(getline(input, newLine)){
        line = (char*) malloc(sizeof(char) * newLine.length() + 1);
        strcpy(line, newLine.c_str());
        if(line[0] == '#' || line[0] == '\n'){
            lineReader();
        }
    }
    else
    {
        line = "";
    }  
}

void readIORequests(){
    do{
        lineReader();
        if(line != ""){
            IOrequest request;
            request.arrivalTime = atoi(strtok(line, delimiter));
            request.track = atoi(strtok(NULL, delimiter));
            io_requests.push_back(request);
        }
    }while(line != "");
}

void set_OptionalTraces(char * options){
    for(int i=0; options[i] != '\0'; i++){
        switch (options[i])
        {
        case 'v':
            v_trace = true;
            break;
        case 'q':
            q_trace = true;
            break;
        case 'f':
            f_trace = true;
            break;
        default:
            break;
        }
    }
}

class Scheduler{
    protected:
    int timestamp = 0, totalMovement = 0, max_waitTime = 0, current_track = 0, next_arrival = 0;
    int requestSize;
    double avg_turnaround = 0.0, avg_waittime = 0.0;
    bool upwardMovement = true;
    list<IOrequest *> IO_queue;
    list<IOrequest *> active_queue;
    IOrequest * current_request = NULL;

    virtual IOrequest * strategy() = 0;

    bool getNextRequest(){
        bool present = false;
        if((!IO_queue.empty()) || (!active_queue.empty())){
            present = true;
            current_request = strategy();
            (*current_request).startTime = timestamp;
            if((*current_request).track < current_track){
                upwardMovement = false;
            }
            else if((*current_request).track > current_track){
                upwardMovement = true;
            }
        }
        return present;
    }

    void addTo_IOQueue(IOrequest * new_request){
        IO_queue.push_back(new_request);
    }

    void summary(){
        int wait;
        for(int i=0; i<requestSize; i++){
            printf("%5d: %5d %5d %5d\n",i, io_requests[i].arrivalTime, io_requests[i].startTime, io_requests[i].endTime);
            wait = io_requests[i].startTime - io_requests[i].arrivalTime;
            avg_waittime += wait;
            avg_turnaround += io_requests[i].turnaroundTime;
            if(max_waitTime < wait){
                max_waitTime = wait;
            }
        }
        avg_waittime /= requestSize;
        avg_turnaround /= requestSize;
        printf("SUM: %d %d %.2lf %.2lf %d\n",--timestamp, totalMovement, avg_turnaround, avg_waittime, max_waitTime);
    }

    bool hasRequest(){
        bool requestPresent = false;
        if((!IO_queue.empty()) || (current_request != NULL) || (!active_queue.empty()) || (next_arrival < requestSize)){
            requestPresent = true;
        }
        return requestPresent;
    }

    public:
    void simulate(){
        requestSize = io_requests.size();
        while (hasRequest())
        {
            if((next_arrival < requestSize) && (io_requests[next_arrival].arrivalTime == timestamp)){
                addTo_IOQueue(&io_requests[next_arrival]);
                next_arrival++;
            }

            while((current_request == NULL) || ((*current_request).track == current_track)){
                if(current_request == NULL){
                    if(!getNextRequest()){
                        break;
                    }
                }
                else{
                    (*current_request).endTime = timestamp;
                    (*current_request).turnaroundTime = timestamp - (*current_request).arrivalTime;
                    current_request = NULL;
                }
            }
            
            if(current_request != NULL){
                if(upwardMovement){
                    current_track++; 
                }
                else{
                    current_track--;
                }
                totalMovement++;
            }
            timestamp++;
        }
        summary();
    }
};

class FIFO: public Scheduler{
    IOrequest * strategy(){
        IOrequest * request = IO_queue.front();
        IO_queue.pop_front();
        return request;
    }
};

class SSTF: public Scheduler{
    IOrequest * strategy(){
        IOrequest * request;
        int shortestTime = INT_MAX;
        int diff;
        list<IOrequest*>::iterator i;
        for(i=IO_queue.begin(); i!=IO_queue.end(); ++i){
            diff = abs((*(*i)).track - current_track);
            if(diff < shortestTime){
                shortestTime = diff;
                request = (*i);
            }
        }
        IO_queue.remove(request);
        return request;
    }
};

class LOOK: public Scheduler{
    IOrequest * strategy(){
        IOrequest * request = NULL;
        IOrequest * request_oppositeDir;
        int shortestTime = INT_MAX;
        int shortestTime_oppositeDir = INT_MAX;
        int diff;
        list<IOrequest*>::iterator i;
        for(i=IO_queue.begin(); i!=IO_queue.end(); ++i){
            if(upwardMovement){
                diff = (*(*i)).track - current_track;
            }
            else{
                diff = current_track - (*(*i)).track;
            }
            if((diff >= 0) && (diff < shortestTime)){
                shortestTime = diff;
                request = (*i);
            }
            else if((diff < 0) && (abs(diff) < shortestTime_oppositeDir)){
                shortestTime_oppositeDir = abs(diff);
                request_oppositeDir = (*i);
            }
        }
        if(request == NULL){
            request = request_oppositeDir;
        }
        IO_queue.remove(request);
        return request;
    }
};

class CLOOK: public Scheduler{
    IOrequest * strategy(){
        IOrequest * request = NULL;
        IOrequest * request_oppositeDir;
        int shortestTime = INT_MAX;
        int shortestTime_oppositeDir = INT_MAX;
        int diff;
        list<IOrequest*>::iterator i;
        for(i=IO_queue.begin(); i!=IO_queue.end(); ++i){
            diff = (*(*i)).track - current_track;
            if((diff >= 0) && (diff < shortestTime)){
                shortestTime = diff;
                request = (*i);
            }
            else if((diff < 0) && (diff < shortestTime_oppositeDir)){
                shortestTime_oppositeDir = diff;
                request_oppositeDir = (*i);
            }
        }
        if(request == NULL){
            request = request_oppositeDir;
        }
        IO_queue.remove(request);
        return request;
    }
};

class FLOOK: public Scheduler{
    IOrequest * strategy(){
        if(active_queue.empty()){
            active_queue = IO_queue;
            IO_queue.clear();
        }
        IOrequest * request = NULL;
        IOrequest * request_oppositeDir;
        int shortestTime = INT_MAX;
        int shortestTime_oppositeDir = INT_MAX;
        int diff;
        list<IOrequest*>::iterator i;
        for(i=active_queue.begin(); i!=active_queue.end(); ++i){
            if(upwardMovement){
                diff = (*(*i)).track - current_track;
            }
            else{
                diff = current_track - (*(*i)).track;
            }
            if((diff >= 0) && (diff < shortestTime)){
                shortestTime = diff;
                request = (*i);
            }
            else if((diff < 0) && (abs(diff) < shortestTime_oppositeDir)){
                shortestTime_oppositeDir = abs(diff);
                request_oppositeDir = (*i);
            }
        }
        if(request == NULL){
            request = request_oppositeDir;
        }
        active_queue.remove(request);
        return request;
    }
};

int main(int argc, char ** inputParam){
    /*argc = 3;
    inputParam[2] = "lab4_assign/input7";
    inputParam[1] = "-ss";*/
    char * inputFile = inputParam[argc - 1];
    char scheduler_type;
    for(int i=0; i<(argc-1); i++){
        switch (inputParam[i][1])
        {
        case 'v':
            v_trace = true;
            break;
        case 'q':
            q_trace = true;
            break;
        case 'f':
            f_trace = true;
            break;
        case 's':
            scheduler_type = inputParam[i][2];
            break;
        default:
            break;
        }
    }
    input.open(inputFile);
    readIORequests();
    input.close();
    if(scheduler_type == 'i'){
        FIFO scheduler;
        scheduler.simulate();
    }
    else if(scheduler_type == 'j'){
        SSTF scheduler;
        scheduler.simulate();
    }
    else if(scheduler_type == 's'){
        LOOK scheduler;
        scheduler.simulate();
    }
    else if(scheduler_type == 'c'){
        CLOOK scheduler;
        scheduler.simulate();
    }
    else{
        FLOOK scheduler;
        scheduler.simulate();
    }
    return 0;
}