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
#define page_table_size 64
#define max_frames 128

ifstream input;
char * delimiter = " ";
char * line = "";
int randomCount, frameSize;
bool O_trace = false;
bool P_trace = false;
bool F_trace = false;
bool S_trace = false;

struct frame{
    int frameId;
    int processId = -1;
    int pte;
    int freeOrder = 0;
    unsigned int age = 0;
} frame_table[max_frames];
struct page_table_entry{
    unsigned int present:1;
    unsigned int write_protect:1;
    unsigned int modified:1;
    unsigned int refered:1;
    unsigned int pagedOut:1;
    unsigned int frameNo:7;
    unsigned int unavailable:1;
    unsigned int fileMapped:1;
    unsigned int swapped:1;
};
struct virtual_memory_area{
    int start;
    int end;
    bool writeProtected;
    bool fileMapped;
};
struct process{
    page_table_entry page_table[page_table_size];
    vector<virtual_memory_area> vmas;
    int segv = 0;
    int segprot = 0;
    int unmap = 0;
    int map = 0;
    int pagein = 0;
    int pageout = 0;
    int zero = 0;
    int fin = 0;
    int fout = 0;
    int read_wirte = 0;
    int context_switch = 0;
    int exit = 0;
};

vector<process> processList;
vector<int> randomValues;

void set_OptionalTraces(char * options){
    for(int i=0; options[i] != '\0'; i++){
        switch (options[i])
        {
        case 'O':
            O_trace = true;
            break;
        case 'P':
            P_trace = true;
            break;
        case 'F':
            F_trace = true;
            break;
        case 'S':
            S_trace = true;
            break;
        default:
            break;
        }
    }
}

void setRandomValues(char * file){
    input.open(file);
    string newLine;
    getline(input, newLine);
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

int getFrameSize(char * c){
    int size;
    c[0] = '0';
    c[1] = '0';
    size = atoi(c);
    return size;
}

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

virtual_memory_area getVMA(){
    virtual_memory_area VMA;
    VMA.start = atoi(strtok(line, delimiter));
    VMA.end = atoi(strtok(NULL, delimiter));
    VMA.writeProtected = atoi(strtok(NULL, delimiter));
    VMA.fileMapped = atoi(strtok(NULL, delimiter));
    return VMA;
}

void readProcesses(){
    lineReader();
    int no_processes = atoi(line);
    for(int i=0; i<no_processes; i++){
        process currentProcess;
        lineReader();
        int no_vmas = atoi(line);
        for(int j=0; j<no_vmas; j++){
            lineReader();
            currentProcess.vmas.push_back(getVMA());
        }
        for(int i=0; i<page_table_size; i++){
            currentProcess.page_table[i] = {0,0,0,0,0,0,0,0,0};
        }
        processList.push_back(currentProcess);
    }
}

class Pager{
    protected:
    char operation;
    int vpage, no_instr=0;
    protected:
    virtual frame* select_victim_frame() = 0;

    frame *allocate_frame_from_free_list(){
        frame *selectedFrame = NULL;
        int minOrder = max_frames;
        for(int i=0; i<frameSize; i++){
            frame_table[i].frameId = i;
            if((frame_table[i].processId == -1) && (frame_table[i].freeOrder < minOrder)){
                selectedFrame = &frame_table[i];
                minOrder = frame_table[i].freeOrder;
                frame_table[i].age = 0;
            }
        }
        if (selectedFrame != NULL) (*selectedFrame).freeOrder = max_frames;
        return selectedFrame;
    }

    frame *get_frame() {
        frame *selectedFrame = allocate_frame_from_free_list();
        if (selectedFrame == NULL) selectedFrame = select_victim_frame();
        return selectedFrame;
    }

    bool get_next_instruction(){
        bool hasInstr = false;
        lineReader();
        if(line != ""){
            hasInstr = true;
            this->operation = strtok(line, delimiter)[0];
            this->vpage = atoi(strtok(NULL, delimiter));
            if(O_trace) cout<<no_instr<<": ==> "<<this->operation<<" "<<this->vpage<<endl;
            no_instr++;
        }
        return hasInstr;
    }

    bool pte_present(page_table_entry *pte, vector<virtual_memory_area> vmas){
        if((*pte).unavailable){
            return false;
        }
        else if((*pte).pagedOut){
            (*pte).present = 1;
            return true;
        }
        bool present = false;
        int size = vmas.size();
        for(int i=0; i<size; i++){
            if((this->vpage >= vmas[i].start) && (this->vpage <= vmas[i].end)){
                present = true;
                (*pte).present = 1;
                (*pte).write_protect = vmas[i].writeProtected;
                (*pte).pagedOut = 1;
                (*pte).fileMapped = vmas[i].fileMapped;
                break;
            }
        }
        if(!(*pte).present){
            (*pte).unavailable = 1;
        }
        return present;
    }

    void unmap_frame(frame *newframe){
        if(O_trace) cout<<" UNMAP "<<(*newframe).processId<<":"<<(*newframe).pte<<endl;
        processList[(*newframe).processId].unmap++;
        if(processList[(*newframe).processId].page_table[(*newframe).pte].modified){
            if(processList[(*newframe).processId].page_table[(*newframe).pte].fileMapped){
                if(O_trace) cout<<" FOUT"<<endl;
                processList[(*newframe).processId].fout++;
            }
            else{
                processList[(*newframe).processId].page_table[(*newframe).pte].swapped = 1;
                if(O_trace) cout<<" OUT"<<endl;
                processList[(*newframe).processId].pageout++;
            }
            processList[(*newframe).processId].page_table[(*newframe).pte].modified = 0;
        }
        processList[(*newframe).processId].page_table[(*newframe).pte].present = 0;
    }

    void map_frame(frame *newframe, int index, page_table_entry *pte, process *current_process){
        if((*pte).fileMapped){
            if(O_trace) cout<<" FIN"<<endl;
            (*current_process).fin++;
        }
        else if((*pte).swapped){
            if(O_trace) cout<<" IN"<<endl;
            (*current_process).pagein++;
        }
        else{
            if(O_trace) cout<<" ZERO"<<endl;
            (*current_process).zero++;
        }
        (*newframe).processId = index;
        (*newframe).pte = this->vpage;
        (*pte).frameNo = (*newframe).frameId;
        if(O_trace) cout<<" MAP "<<(*newframe).frameId<<endl;
        (*current_process).map++;
    }

    void update_pte(page_table_entry *pte, process *current_process){
        (*pte).refered = 1;
        (*current_process).read_wirte++;
        if(this->operation == 'w'){
            if(!(*pte).write_protect){
                (*pte).modified = 1;
            }
            else{
                if(O_trace) cout<<" SEGPROT"<<endl;
                (*current_process).segprot++;
            }
        }
    }

    void exit_unmap(int index, process *current_process){
        for(int i=0; i<page_table_size; i++){
            if((*current_process).page_table[i].present){
                if(O_trace) cout<<" UNMAP "<<index<<":"<<i<<endl;
                (*current_process).unmap++;
                if((*current_process).page_table[i].modified){
                    if((*current_process).page_table[i].fileMapped){
                        if(O_trace) cout<<" FOUT"<<endl;
                        (*current_process).fout++;
                    }
                    (*current_process).page_table[i].modified = 0;
                }
                frame_table[(*current_process).page_table[i].frameNo].processId = -1;
                frame_table[(*current_process).page_table[i].frameNo].freeOrder = i;
                (*current_process).page_table[i].present = 0;
            }
            (*current_process).page_table[i].swapped = 0;
        }
    }

    void summary(){
        if(P_trace){
            for(int i=0; i<processList.size(); i++){
                cout<<"PT["<<i<<"]:";
                for(int j=0; j<page_table_size; j++){
                    cout<<" ";
                    if(processList[i].page_table[j].present){
                        char bits[3];
                        bits[0] = processList[i].page_table[j].refered ? 'R' : '-';
                        bits[1] = processList[i].page_table[j].modified ? 'M' : '-';
                        bits[2] = processList[i].page_table[j].swapped ? 'S' : '-';
                        cout<<j<<":"<<bits;
                    }
                    else if(processList[i].page_table[j].swapped){
                        cout<<"#";
                    }
                    else{
                        cout<<"*";
                    }
                }
                cout<<endl;
            }
        }
        
        if(F_trace){
            cout<<"FT:";
            for(int i=0; i<frameSize; i++){
                if(frame_table[i].processId == -1){
                    cout<<" *";
                }
                else{
                    cout<<" "<<frame_table[i].processId<<":"<<frame_table[i].pte;
                }
            }
            cout<<endl;
        }
        
        int context_switch=0, exit=0;
        unsigned long long total_cost=0;
        for(int i=0; i<processList.size(); i++){
            cout<<"PROC["<<i<<"]: U="<<processList[i].unmap<<" M="<<processList[i].map<<" I="<<processList[i].pagein;
            cout<<" O="<<processList[i].pageout<<" FI="<<processList[i].fin<<" FO="<<processList[i].fout;
            cout<<" Z="<<processList[i].zero<<" SV="<<processList[i].segv<<" SP="<<processList[i].segprot<<endl;

            context_switch += processList[i].context_switch;
            exit += processList[i].exit;
            total_cost += processList[i].segv * 240;
            total_cost += processList[i].segprot * 300;
            total_cost += processList[i].map * 400;
            total_cost += processList[i].unmap * 400;
            total_cost += processList[i].pagein * 3000;
            total_cost += processList[i].pageout * 3000;
            total_cost += processList[i].zero * 150;
            total_cost += processList[i].fin * 2500;
            total_cost += processList[i].fout * 2500;
            total_cost += processList[i].read_wirte * 1;
            total_cost += processList[i].context_switch * 121;
            total_cost += processList[i].exit * 175;
        }

        if(S_trace) cout<<"TOTALCOST "<<no_instr<<" "<<context_switch<<" "<<exit<<" "<<total_cost;
    }

    public:
    void simulate(){
        process *current_process = NULL;
        int current_process_index;
        while (get_next_instruction()) {
            if(this->operation == 'c'){
                current_process = &processList[this->vpage];
                (*current_process).context_switch++;
                current_process_index = this->vpage;
            }
            else if(this->operation == 'e'){
                (*current_process).exit++;
                if(O_trace) cout<<"EXIT current process "<<current_process_index<<endl;
                exit_unmap(current_process_index, current_process);
                current_process = NULL;
            }
            else{
                (*current_process).read_wirte;
                page_table_entry *pte = &(*current_process).page_table[this->vpage];
                frame *newframe;
                if ( ! (*pte).present) {
                    if(pte_present(pte, (*current_process).vmas)){
                        newframe = get_frame();
                        if((*newframe).processId != -1){
                            unmap_frame(newframe);
                        }
                        map_frame(newframe, current_process_index, pte, current_process);
                    }
                    else{
                        if(O_trace) cout<<" SEGV"<<endl;
                        (*current_process).read_wirte++;
                        (*current_process).segv++;
                        continue;
                    }
                }
                update_pte(pte, current_process);
            }
        }
        summary();
    }
};

class FIFO: public Pager{
    int hand=0;
    public:
    frame* select_victim_frame(){
        frame * victim;
        victim = &frame_table[hand];
        hand++;
        if(hand == frameSize){
            hand = 0;
        }
        return victim;
    }
};

class Clock: public Pager{
    int hand=0;
    public:
    frame* select_victim_frame(){
        frame * victim;
        while(processList[frame_table[hand].processId].page_table[frame_table[hand].pte].refered){
            processList[frame_table[hand].processId].page_table[frame_table[hand].pte].refered = 0;
            hand++;
            if(hand == frameSize){
                hand = 0;
            }
        }
        victim = &frame_table[hand];
        hand++;
        if(hand == frameSize){
            hand = 0;
        }
        return victim;
    }
};

class Aging: public Pager{
    int hand=0;
    void update_ages(){
        for(int i=0; i<frameSize; i++){
            frame_table[i].age = frame_table[i].age >> 1;
            if(processList[frame_table[i].processId].page_table[frame_table[i].pte].refered){
                frame_table[i].age = (frame_table[i].age | 0x80000000);
                processList[frame_table[i].processId].page_table[frame_table[i].pte].refered = 0;
            }
        }
    }
    
    public:

    frame* select_victim_frame(){
        frame * victim;
        int currentHand = hand-1;
        update_ages();
        unsigned int minAge = UINT_MAX;
        for(int i=0; i<frameSize; i++){
            currentHand++;
            if(currentHand >= frameSize){
                currentHand = 0;
            }
            if(frame_table[currentHand].age < minAge){
                hand = currentHand;
                minAge = frame_table[hand].age;
            }
        }
        frame_table[hand].age = 0;
        victim = &frame_table[hand++];
        return victim;
    }
};

class Working_set: public Pager{
    int hand = 0;
    int tau = 49;
    int ages[max_frames];
    public: 

    Working_set(){
        for(int i=0; i<frameSize; i++){
            ages[i] = 0;
        }
    }

    frame* select_victim_frame(){
        int currentHand = hand-1;
        int maxAge = 0;
        int age;
        frame * victim;
        bool reset = false;
        for(int i=0; i<frameSize; i++){
            currentHand++;
            if(currentHand >= frameSize){
                currentHand = 0;
            }
            age = no_instr - ages[currentHand];
            if(processList[frame_table[currentHand].processId].page_table[frame_table[currentHand].pte].refered){
                ages[currentHand] = no_instr;
                processList[frame_table[currentHand].processId].page_table[frame_table[currentHand].pte].refered = 0;
            }
            else{
                if(age > tau){
                    hand = currentHand;
                    break;
                }
                else if(age > maxAge){
                    hand = currentHand;
                    maxAge = age;
                }
            }

        }
        ages[hand] = no_instr;
        victim = &frame_table[hand++];
        return victim;
    }
};

class NRU: public Pager{
    int hand = 0;
    int reset_instr = 0;
    public:
    frame* select_victim_frame(){
        int currentHand = hand-1;
        int minCost = 4;
        int cost;
        frame * victim;
        bool reset = false;
        if((no_instr - reset_instr) >= 50){
            reset = true;
            reset_instr = no_instr;
        }
        for(int i=0; i<frameSize; i++){
            currentHand++;
            if(currentHand >= frameSize){
                currentHand = 0;
            }
            cost = (2 * processList[frame_table[currentHand].processId].page_table[frame_table[currentHand].pte].refered) + processList[frame_table[currentHand].processId].page_table[frame_table[currentHand].pte].modified;
            if(reset){
                processList[frame_table[currentHand].processId].page_table[frame_table[currentHand].pte].refered = 0;
            }
            if(cost < minCost){
                hand = currentHand;
                if((cost == 0) && (!reset)){
                    break;
                }
                minCost = cost;
            }
        }
        victim = &frame_table[hand++];
        return victim;
    }
};

class Random: public Pager{
    int randomOfset = 0;
    int myrandom(){
        if(randomOfset >= randomCount){
            randomOfset = 0;
        }
        return (randomValues[randomOfset++] % frameSize); 
    }

    public:
    frame* select_victim_frame(){
        frame * victim;
        int hand = myrandom();
        victim = &frame_table[hand];
        return victim;
    }    
};

int main(int argc, char ** inputParam){
    char * inputFile = inputParam[argc - 2];
    char * randomValueFile = inputParam[argc - 1];
    setRandomValues(randomValueFile);
    char replacementAlgorithm;
    for(int i=1; i<(argc-2);i++){
        switch (inputParam[i][1])
        {
        case 'a':
            replacementAlgorithm = inputParam[i][2];
            break;
        case 'f':
            frameSize = getFrameSize(inputParam[i]);
            break;
        case 'o':
            set_OptionalTraces(inputParam[i]);
            break;
        default:
            break;
        }
    }
    
    input.open(inputFile);
    readProcesses();
    if(replacementAlgorithm == 'f'){
        FIFO simulator;
        simulator.simulate();
    }
    else if(replacementAlgorithm == 'c'){
        Clock simulator;
        simulator.simulate();
    }
    else if(replacementAlgorithm == 'e'){
        NRU simulator;
        simulator.simulate();
    }
    else if(replacementAlgorithm == 'r'){
        Random simulator;
        simulator.simulate();
    }
    else if(replacementAlgorithm == 'a'){
        Aging simulator;
        simulator.simulate();
    }
    else{
        Working_set simulator;
        simulator.simulate();
    }
    input.close();
    return 0;
}