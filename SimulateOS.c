#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <stdlib.h>


// Function to list background jobs
void list_background_jobs(int *bg_pids,char **bg_commands,int count){
        printf("Total Background jobs: %d\n",count);
        for(int i=0;i<count;i++){
                printf("%d: %s\n",bg_pids[i],bg_commands[i]);
        }

}

int main(int argc,char* argv[]){
        char cwd[1024];
        char hostname[1024];
        int bg_pids[1024];
        char input[1024];
        char *bg_commands[1024];
        int bg_count=0;

        while(1){
        getcwd(cwd,sizeof(cwd));
        char *user=getlogin();
        gethostname(hostname,sizeof(hostname));
        printf("%s@%s:%s > ",user,hostname,cwd);

        // Read user input
        fgets(input,sizeof(input),stdin);
        input[strcspn(input,"\n")]='\0';

        // Exit if user types 'exit'
        if(strcmp(input,"exit")==0){
                break;
        }

        char new_str[sizeof(input)];
        strcpy(new_str,input);

        if(strncmp(input,"bg ",3)==0){
                // Background process handling
                char *bg_input=input+3;
                char *bg_str=strtok(bg_input," ");
                char *bgstr[128];
                for(int i=0;i<sizeof(bgstr);i++){
                        bgstr[i]=NULL;
                }


                for(int i=0;bg_str!=NULL;i++){
                        bgstr[i]=malloc(strlen(bg_str)+1);
                        strcpy(bgstr[i],bg_str);
                        bg_str=strtok(NULL," ");
                }


                if(strcmp(bgstr[0],"cd")==0){
                        chdir(bgstr[1]);
                }else{
                        pid_t pid=fork();
                        if(pid==0){
                                execvp(bgstr[0],bgstr);
                                perror("EXECVP FOR  BG");
                                exit(1);
                        }else if(pid>0){
                                bg_pids[bg_count]=pid;
                                bg_commands[bg_count]=malloc(strlen(input)+1);
                                strcpy(bg_commands[bg_count],input);
                                bg_count++;
                        }else{
                                perror("FORK FOR BG");
                        }
                }
        }else if(strcmp(input,"bglist")==0){
                // List background jobs
                list_background_jobs(bg_pids,bg_commands,bg_count);

        }else{
            // Other commands
                char *str[128];
                for(int i=0;i<sizeof(str);i++){
                        str[i]=NULL;
                }
                char *_str=strtok(input," ");
                for(int i=0;_str!=NULL;i++){
                        str[i]=malloc(strlen(_str)+1);
                        strcpy(str[i],_str);
                        _str=strtok(NULL," ");
                }

                if(strcmp(str[0],"cd")==0){
                        chdir(str[1]);
                }else{
                        pid_t pid=fork();
                        if(pid==0){
                                execvp(str[0],str);
                                perror("execvp for notbg");
                                exit(1);
                        }else if(pid>0){
                                wait(NULL);
                        }else{
                                perror("fork for notbg");
                        }
                }
        }

        // Handle terminated background jobs
        int status;
        pid_t terminated_pid;
        while((terminated_pid=waitpid(-1,&status,WNOHANG))>0){
                for(int i=0;i<bg_count;i++){
                        if(bg_pids[i]==terminated_pid){
                                printf("%d: %s has terminated.\n",bg_pids[i],bg_commands[i]);
                                free(bg_commands[i]);
                                for(int j=i;j<bg_count-1;j++){
                                        bg_pids[j]=bg_pids[j+1];
                                        bg_commands[j]=bg_commands[j+1];
                                }
                                bg_count--;
                                break;
                        }
                }
        }

        }
        return 0;
}