#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <iomanip>
#include <ctype.h>
#include <stdlib.h>

using namespace std;

ifstream input;
char * line = "";
char * delimiter = " \t";
char * fileName = "lab1samples//input-1";
int ofset;
char * token;
char * lastToken = "";
int lineCount = 0;
int baseAddress;
int pgmLength;
int moduleCount;
int defCount;
struct savedSymbol
{
    char * sym;
    int value;
    string message;
};
vector<savedSymbol> symbolTable;
vector<char*> useList;
struct usedSymbols
{
    char * sym;
    int moduleNo;
};
vector<usedSymbols> moduleUsedSymbols;
struct memory
{
    int instr;
    string message;
    vector<usedSymbols> warnings;
};
vector<memory> memoryMap;
struct defSymbols
{
    char * sym;
    int moduleNo;
};
vector<defSymbols> definedSymbols;
struct tooLargeDeclarations{
    int moduleNo, size, maxSize;
    char * sym;
};
vector<tooLargeDeclarations> largeSymbolWarnings;

string handleError(int code, char * symbol = ""){
    string errorMessage[] = {
        " Error: Absolute address exceeds machine size; zero used",
        " Error: Relative address exceeds module size; zero used",
        " Error: External address exceeds length of uselist; treated as immediate",
        " Error: s is not defined; zero used",
        " Error: This variable is multiple times defined; first value used",
        " Error: Illegal immediate value; treated as 9999",
        " Error: Illegal opcode; treated as 9999"
    };
    if(code == 3){
        errorMessage[3].replace(8,1,symbol);
    }
    return(errorMessage[code]);
}

void handleSyntaxError(int code){
    static char* errorMessage[] = {
        "NUM_EXPECTED", // Number expect
        "SYM_EXPECTED", // Symbol Expected
        "ADDR_EXPECTED", // Addressing Expected which is A/E/I/R
        "SYM_TOO_LONG", // Symbol Name is too long
        "TOO_MANY_DEF_IN_MODULE", // > 16
        "TOO_MANY_USE_IN_MODULE", // > 16
        "TOO_MANY_INSTR" // total num_instr exceeds memory size (512)
        };

    cout<<"Parse Error line "<<lineCount<<" offset "<<ofset<<": "<<errorMessage[code];
    exit(0);
}

void insertSymbol(char * c, int i, char * m){
    bool duplicate = false;
    int size = symbolTable.size();
    for(int i=0; i<size;i++){
        if(!strcmp(symbolTable[i].sym, c)){
            symbolTable[i].message = handleError(4);
            duplicate = true;
            break;
        }
    }
    if(!duplicate){
        struct savedSymbol newSymbol;
        newSymbol.sym = c;
        newSymbol.value = i;
        newSymbol.message = m;
        symbolTable.push_back(newSymbol);
        struct defSymbols unusedSymbol;
        unusedSymbol.sym = c;
        unusedSymbol.moduleNo = moduleCount;
        definedSymbols.push_back(unusedSymbol);
    }
}

void displaySymbolTable(){
    cout<<"Symbol Table"<<endl;
    int size = symbolTable.size();
    for(int i=0; i<size;i++){
        cout<<symbolTable[i].sym<<"="<<symbolTable[i].value<<symbolTable[i].message<<endl;
    }
}

void insertMemory(int i, string m){
    struct memory newMemory;
    newMemory.instr = i;
    newMemory.message = m;
    memoryMap.push_back(newMemory);
}

void displayMemoryMap(){
    cout<<"Memory Map"<<endl;
    int size = memoryMap.size();
    for(int i=0; i<size; i++){
        cout<<setfill('0')<<setw(3)<<i<<": "<<setfill('0')<<setw(4)<<memoryMap[i].instr<<memoryMap[i].message<<endl;
        if(memoryMap[i].warnings.size() != 0){
            int warnings = memoryMap[i].warnings.size();
            for(int j=0; j<warnings; j++){
                cout<<"Warning: Module "<<memoryMap[i].warnings[j].moduleNo<<": "<<memoryMap[i].warnings[j].sym<<" appeared in the uselist but was not actually used\n";
            }
        }
    }
}

void usedSymbol(char * c){
    int size = definedSymbols.size();
    for(int i=0; i<size; i++){
        if(!strcmp(definedSymbols[i].sym, c)){
            definedSymbols.erase(definedSymbols.begin() + i);
            size--;
        }
    }
}

void displayUnusedDefs(){
    int size = definedSymbols.size();
    for(int i=0; i<size; i++){
        cout<<endl;
        cout<<"Warning: Module "<<definedSymbols[i].moduleNo<<": "<<definedSymbols[i].sym<<" was defined but never used";
    }
}

void symbolReferred(char * c){
    int size = moduleUsedSymbols.size();
    for(int i=0; i<size; i++){
        if(!strcmp(moduleUsedSymbols[i].sym, c)){
            moduleUsedSymbols.erase(moduleUsedSymbols.begin() + i);
            size--;
        }
    }
}

void addLargeDefWarning(int module, char* sym, int size, int maxSize){
    struct tooLargeDeclarations largeSymbol;
    largeSymbol.moduleNo = module;
    largeSymbol.sym = sym;
    largeSymbol.size = size;
    largeSymbol.maxSize = maxSize;
    largeSymbolWarnings.push_back(largeSymbol);
}

void displayLargeDefWarnings(){
    int size = largeSymbolWarnings.size();
    for(int i=0; i<size; i++){
        cout<<"Warning: Module "<<largeSymbolWarnings[i].moduleNo<<": "<<largeSymbolWarnings[i].sym<<" too big "<<largeSymbolWarnings[i].size<<" (max="<<largeSymbolWarnings[i].maxSize<<") assume zero relative\n";
    }
}

void checklargeDefination(int moduleSize){
    for(int i = (symbolTable.size() - defCount); i<symbolTable.size(); i++){
        if((symbolTable[i].value-baseAddress) > (moduleSize - 1)){
            addLargeDefWarning(moduleCount, symbolTable[i].sym, symbolTable[i].value-baseAddress, moduleSize - 1);
            symbolTable[i].value = baseAddress;
        }
    }
}

int readLine(){
    string newLine;
    if(getline(input, newLine)){
        line = (char*) malloc(sizeof(char) * newLine.length() + 1);
        strcpy(line, newLine.c_str());
        lineCount++;
        lastToken = "";
        return true;
    }
    else{
        return false;
    }
}

void getToken(bool isNewLine){
    if(isNewLine || (line == "")){
        token = strtok(line,delimiter);
    }
    else{
        token = strtok(NULL,delimiter);
    }

    if(token == NULL){
        if(readLine()){
            getToken(true);
        }
        else{
            token = "";
            if(strlen(lastToken)){
                ofset += strlen(lastToken);
            } 
        }
    }
    else{
        lastToken = token;
        ofset = token-line+1;
    }
}

int readInt(int pass, bool required){
    int num;
    getToken(false);
    if(pass == 1){
        for(int i=0; token[i]!='\0'; i++){
            if(!isdigit(token[i])){
                handleSyntaxError(0);
            }
        }
    }
    if(token == ""){
        if(required){
            handleSyntaxError(0);
        }
        else
        {
            return 0;
        }
    }
    num = atoi(token);
    return num;
}

void readSymbol(int pass){
    getToken(false);
    if(pass == 1){
        int i;
        if((strlen(token)==0) || (!isalpha(token[0]))){
            handleSyntaxError(1);
        }
        for(i=1; token[i]!='\0'; i++){
            if(!isalnum(token[i])){
                handleSyntaxError(1);
            }
        }
        if(i>16){
            handleSyntaxError(3);
        }
    }
}

void readIAER(int pass){
    getToken(false);
    if(pass == 1){
        if(strcmp(token,"A") && strcmp(token,"E") && strcmp(token,"I") && strcmp(token,"R")){
            handleSyntaxError(2);
        }
    }
}

void instrTransform(char addMode, int instr){
    string message = "";
    if((addMode == 'I') && (instr >= 10000)){
        instr = 9999;
        message = handleError(5);
    }
    else if(instr >= 10000){
        instr = 9999;
        message = handleError(6);
    }
    
    else if((addMode == 'A') && ((instr%1000) >= 512)){
        instr = (instr/1000)*1000;
        message = handleError(0);
    }
    else if(addMode == 'R'){
        if((instr%1000) > pgmLength){
            instr = (instr/1000)*1000;
            message = handleError(1);
        }
        instr += baseAddress - pgmLength;
    }
    else if(addMode == 'E'){
        int symbolIndex = instr%1000;
        int opcode = instr/1000;
        if(symbolIndex >= useList.size()){
            message = handleError(2);
        }
        else{
            char * symbolUsed = useList[symbolIndex];
            symbolReferred(symbolUsed);
            bool symFound = false;
            int size = symbolTable.size();
            for(int i=0; i<size; i++){
                if(!strcmp(symbolTable[i].sym,symbolUsed)){
                    instr = (opcode*1000) + (symbolTable[i].value);
                    symFound = true;
                    break;
                }
            }
            if(!symFound){
                instr = opcode*1000;
                message = handleError(3, symbolUsed);
            }
        } 
    }
    insertMemory(instr, message);
}

void parseDefList(int pass){
    defCount = readInt(pass, false);
    if(pass == 1 && defCount>16){
        handleSyntaxError(4);
    }
    for(int i=0; i<defCount; i++){
        readSymbol(pass);
        char * symbol = token;
        int val = readInt(pass, true);
        val += baseAddress;
        if(pass == 1){
            insertSymbol(symbol, val, "");
        }
    }
}

void parseUseList(int pass){
    int useCount = readInt(pass, false);
    if(pass == 1 && useCount>16){
        handleSyntaxError(5);
    }
    useList.clear();
    moduleUsedSymbols.clear();
    for(int i=0; i<useCount; i++){
        readSymbol(pass);
        if(pass == 2){
            usedSymbol(token);
            useList.push_back(token);
            struct usedSymbols symToUse;
            symToUse.sym = token;
            symToUse.moduleNo = moduleCount;
            moduleUsedSymbols.push_back(symToUse);
        }
    }
}

void parsePgmText(int pass){
    pgmLength = readInt(pass, false);
    if(pass == 1){
        checklargeDefination(pgmLength);
    }
    char addMode;
    baseAddress += pgmLength;
    if(pass == 1 && baseAddress>512){
        handleSyntaxError(6);
    }
    for(int i=0; i<pgmLength; i++){
        readIAER(pass);
        addMode = token[0];
        int instruction = readInt(pass, true);
        if(pass == 2){
            instrTransform(addMode, instruction);
        }
    }
    if((pass == 2) && (moduleUsedSymbols.size())){
        memoryMap[memoryMap.size()-1].warnings = moduleUsedSymbols;
    }
}

int main(int argc, char ** inputFile){
    fileName = inputFile[1];
    int pass;
    /* The variable "pass" represents which pass the linker is executing. 
       This value has been referred in the common functions to trigger certain actions depending upon the pass.*/
    for(pass=1; pass<=2; pass++){
        moduleCount = 0;
        baseAddress = 0;
        input.open(fileName);
        while (!input.eof())
        {
            moduleCount++;
            parseDefList(pass);
            parseUseList(pass);
            parsePgmText(pass);
        }
        input.close();
    }
    displayLargeDefWarnings();
    displaySymbolTable();
    cout<<endl;
    displayMemoryMap();
    displayUnusedDefs();
    return 0;
}