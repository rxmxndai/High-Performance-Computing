#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>
#include <errno.h>
#include <stdbool.h>

#define handle_error_en(en, msg) \
  do                             \
  {                              \
    errno = en;                  \
    perror(msg);                 \
    exit(EXIT_FAILURE);          \
  } while (0)

/******************************************************************************
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 2 uppercase
  letters and a 2 digit integer.

  Compile with:
    gcc CrackAZ99.c -pthread -lcrypt -o CrackAZ99

  Execute with:
    ./CrackAZ99 <number_of_threads>
    where number_of_threads should be > 0
*******************************************************************************/

int count = 0; // A counter for combination explored
int threadCount;

int loopCount = 26; // to iterate through the characters `A` to `Z` of the outer for-loop
bool found = false;

struct threadInfo
{
  int start;
  int end;
};

char startChar, endChar;

char *salt_and_encrypted;

sem_t sem;

// Required by lack of standard function in C.
void substr(char *dest, char *src, int start, int length)
{
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All combinations
 that are tried are displayed and when the password is found, #, is put at the
 start of the line. Note that one of the most time consuming operations that
 it performs is the output of intermediate results, so performance experiments
 for this kind of program should not include this. i.e. comment out the printfs.
*/

static void *crack(void *args)
{
  sem_wait(&sem);

  int s;

  s = pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
  if (s != 0)
    handle_error_en(s, "pthread_setcancelstate");

  int x, y, z;   // Loop counters
  char salt[7];  // String used in hashing the password. Need space for \0 // incase you have modified the salt value, then should modifiy the number accordingly
  char plain[7]; // The combination of letters currently being checked // Please modifiy the number when you enlarge the encrypted password.
  char *enc;     // Pointer to the encrypted password

  char reverse; // to convert the ASCII int into ASCII char value

  substr(salt, salt_and_encrypted, 0, 6);

  struct threadInfo *tI = (struct threadInfo *)args;
  int start = tI->start;
  int end = tI->end;

  if (!found)
  {
    char startPoint = start;
    char endPoint = end;
    printf("\nLooping through `%c` to `%c`\n", startPoint, endPoint);

    for (x = start; x <= end; x++)
    {
      reverse = x;
      for (y = 'A'; y <= 'Z'; y++)
      {
        for (z = 0; z <= 99; z++)
        {
          sprintf(plain, "%c%c%02d", reverse, y, z);
          enc = (char *)crypt(plain, salt);
          count++;
          if (strcmp(salt_and_encrypted, enc) == 0)
          {
            printf("\n\n#%-8d%s %s\n\n", count, plain, enc);

            found = true;
            printf("\nEncrypted password found!");
            printf("\nNow exiting in a few seconds...\n");
          }
        }
      }
    }
  }
  else
  {
    // cancel all other threads when the required password has been found
    s = pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    if (s != 0)
      handle_error_en(s, "pthread_setcancelstate");

    // waiting for a second to let the thread cancel
    // printf("\nWaiting for five seconds to let the remaining threads cancel...\n");
    sleep(5);

    // if it reaches to this print statement, it means that the thread is not cancelled
    // printf("\nthreads not cancelled...\n");
  }

  sem_post(&sem);
}

// preparing the slicelist
void prepareSliceList()
{
  int sliceList[threadCount];
  int remainder = loopCount % threadCount;

  void *res;
  int s;

  // to store the sliced/divided number of characters to be processed by each thread
  for (int i = 0; i < threadCount; i++)
  {
    sliceList[i] = loopCount / threadCount;
  }

  // to update the sliced/divided number of characters that each thread
  // has to process without leaving any characters unprocessed/unchecked
  for (int j = 0; j < remainder; j++)
  {
    sliceList[j] = sliceList[j] + 1;
  }

  int startList[threadCount];
  int endList[threadCount];

  /*
   * For threads = 3,
   * first thread will loop through the outer loop processing the characters from A through I i.e., first 9 uppercase characters
   * second thread will loop through the outer loop processing the characters from J through R i.e., other 9 uppercase characters
   * third thread will loop through the outer loop processing the characters from S through Z i.e., the remaining 8 uppercase characters
   *  */
  for (int k = 0; k < threadCount; k++)
  {
    if (k == 0)
    {
      startList[k] = 65; // ASCII value of 'A'
    }
    else
    {
      startList[k] = endList[k - 1] + 1;
    }

    endList[k] = startList[k] + sliceList[k] - 1;

    printf("\nstartList[%d] = %d / `%c`\t\tendList[%d] = %d / `%c`", k, startList[k], (char)startList[k], k, endList[k], (char)endList[k]);
  }

  struct threadInfo threadDetails[threadCount];

  for (int l = 0; l < threadCount; l++)
  {
    threadDetails[l].start = startList[l];
    threadDetails[l].end = endList[l];
  }

  pthread_t thread_id[threadCount];

  sem_init(&sem, 0, 1);

  printf("\n\nCreating threads and checking for a matching hash...\n");

  // Copy and paste the ecrypted password here using EncryptShA512 program
  salt_and_encrypted = "$6$AS$xeLDdLSAyepaRppxN6uk66dzaeHUn7SbQRo1Ac7PZFHtCl.oGxEXu/KCWzCILEsStrf2QkWmO.FulkowaGtoY0";

  printf("\nInput salt_and_encrypted: %s\n", salt_and_encrypted);

  for (int m = 0; m < threadCount; m++)
  {
    s = pthread_create(&thread_id[m], NULL, &crack, &threadDetails[m]);
    if (s != 0)
      handle_error_en(s, "pthread_create");
  }

  for (int n = 0; n < threadCount; n++)
  {
    if (found)
    {
      // printf("\nThreadID: %d is canceling\n", n);

      s = pthread_cancel(thread_id[n]);
      if (s != 0)
        handle_error_en(s, "pthread_cancel");
    }

    s = pthread_join(thread_id[n], &res);

    if (s != 0)
      handle_error_en(s, "pthread_join");

    // if (res == PTHREAD_CANCELED)
    //   printf("\nThreadID: %d was canceled...\n", n);
    // else
    //   printf("\nThreadID: %d was not canceled...\n", n);
  }

  // printf("\nsemaphore destroyed...\n");

  sem_destroy(&sem);
}

// function to display a message explaining what and how arguments should be passed
void inputArguments(char *program_name)
{
  fprintf(stderr, "arguments should be in the order as specified:   %s <number of threads>\n", program_name);
  fprintf(stderr, "where number of threads should be > 0\n");
  exit(0);
}

// getArguments() method created to get the command line arguments
void getArguments(int argc, char *argv[])
{
  if (argc != 2)
  {
    inputArguments(argv[0]);
  }

  threadCount = strtol(argv[1], NULL, 10);

  if (threadCount <= 0)
  {
    inputArguments(argv[0]);
  }
}

int main(int argc, char *argv[])
{
  getArguments(argc, argv);

  prepareSliceList();

  printf("\n%d solutions explored\n", count);
  return 0;
}
