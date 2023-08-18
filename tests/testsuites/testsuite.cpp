/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include "../common.hpp"
#include "testsuite.hpp"

namespace gdrcopy {
    namespace testsuite {
        TestMap *get_test_map()
        {
            static TestMap *_test_map = NULL;
            if (!_test_map)
                _test_map = new TestMap();
            return _test_map;
        }

        void get_all_test_names(std::vector<std::string>& names)
        {
            TestMap *test_map = get_test_map();
            
            names.clear();
            for (TestMap::iterator it = test_map->begin(); it != test_map->end(); ++it)
                names.emplace_back(it->first);
        }

        int run_tests(bool print_summary, std::vector<std::string> tests)
        {
            TestMap *test_map = get_test_map();

            std::vector<std::string> success_tests;
            std::vector<std::string> failed_tests;
            std::vector<std::string> waived_tests;

            for (std::vector<std::string>::iterator it = tests.begin(); it != tests.end(); ++it) {
                if (test_map->find(*it) == test_map->end()) {
                    gdrcopy::test::print_dbg("Error: Encountered unknown test %s\n", *it);
                    return EINVAL;
                }
            }

            for (std::vector<std::string>::iterator it = tests.begin(); it != tests.end(); ++it) {
                test_status_t status = (*test_map)[*it]->run();
                switch (status) {
                    case TEST_STATUS_SUCCESS:
                        success_tests.push_back(*it);
                        break;
                    case TEST_STATUS_FAILED:
                        failed_tests.push_back(*it);
                        break;
                    case TEST_STATUS_WAIVED:
                        waived_tests.push_back(*it);
                        break;
                    default:
                        gdrcopy::test::print_dbg("Error: Unknown test status\n");
                        return EINVAL;
                }
            }

            if (print_summary) {
                printf(
                    "Total: %lu, Passed: %lu, Failed: %lu, Waived: %lu\n",
                    tests.size(),
                    success_tests.size(),
                    failed_tests.size(),
                    waived_tests.size()
                );
                if (failed_tests.size() > 0) {
                    printf("\n");
                    printf("List of failed tests:\n");
                    for (std::vector<std::string>::iterator it = failed_tests.begin(); it != failed_tests.end(); ++it)
                        printf("    %s\n", it->c_str());
                }
                if (waived_tests.size() > 0) {
                    printf("\n");
                    printf("List of waived tests:\n");
                    for (std::vector<std::string>::iterator it = waived_tests.begin(); it != waived_tests.end(); ++it)
                        printf("    %s\n", it->c_str());
                }
            }

            if (failed_tests.size() > 0)
                return TEST_STATUS_FAILED;

            return TEST_STATUS_SUCCESS;
        }

        int run_all_tests(bool print_summary)
        {
            std::vector<std::string> tests;
            get_all_test_names(tests);
            return run_tests(print_summary, tests);
        }

        Test::Test(std::string name) : t_name(name)
        {
            register_test();
        }

        test_status_t Test::run()
        {
            test_status_t status = TEST_STATUS_FAILED;
            const char *testname = t_name.c_str();
            int filedes_0[2];
            int filedes_1[2];
            int read_fd;
            int write_fd;
            ASSERT_NEQ(pipe(filedes_0), -1);
            ASSERT_NEQ(pipe(filedes_1), -1);

            gdrcopy::test::print_dbg("&&&& RUNNING %s\n", testname);
            fflush(stdout);
            fflush(stderr);
            pid_t pid = fork();
            if (pid < 0) {
                gdrcopy::test::print_dbg("Cannot fork\n");
                gdrcopy::test::print_dbg("&&&& FAILED %s\n", testname);
                status = TEST_STATUS_FAILED;
                goto out;
            }                                                                   
            else if (pid > 0){
                int child_exit_status = -EINVAL;
                pid_t group_pid;
                ssize_t read_size;

                close(filedes_0[1]);
                close(filedes_1[0]);
                read_fd = filedes_0[0];
                write_fd = filedes_1[1];

                read_size = read(read_fd, &group_pid, sizeof(group_pid));
                if (read_size != sizeof(group_pid)) {
                    gdrcopy::test::print_dbg("Error in receiving group_pid\n");
                    gdrcopy::test::print_dbg("&&&& FAILED %s\n", testname);
                    status = TEST_STATUS_FAILED;
                    goto out;
                }

                close(read_fd);
                close(write_fd);

                if (waitpid(pid, &child_exit_status, 0) == -1) {
                    gdrcopy::test::print_dbg("waitpid returned an error\n");
                    gdrcopy::test::print_dbg("&&&& FAILED %s\n", testname);
                    status = TEST_STATUS_FAILED;
                    goto out;
                }

                if (!WIFEXITED(child_exit_status)) {
                    gdrcopy::test::print_dbg("Error: The test process got terminated\n");
                    gdrcopy::test::print_dbg("&&&& FAILED %s\n", testname);
                    status = TEST_STATUS_FAILED;
                    goto out;
                }

                child_exit_status = WEXITSTATUS(child_exit_status);
                if (child_exit_status == EXIT_SUCCESS) {
                    gdrcopy::test::print_dbg("&&&& PASSED %s\n", testname);
                    status = TEST_STATUS_SUCCESS;
                }
                else if (child_exit_status == EXIT_WAIVED) {
                    gdrcopy::test::print_dbg("&&&& WAIVED %s\n", testname);
                    status = TEST_STATUS_WAIVED;
                }
                else {
                    gdrcopy::test::print_dbg("&&&& FAILED %s\n", testname);
                    status = TEST_STATUS_FAILED;
                }

                killpg(group_pid, SIGKILL);
                goto out;
            }
            else {
                close(filedes_0[0]);
                close(filedes_1[1]);
                read_fd = filedes_1[0];
                write_fd = filedes_0[1];
                this->run_executor(read_fd, write_fd);
            }

out:
            return status;
        }

        void Test::run_executor(int read_fd, int write_fd)
        {
            int status = 0;
            pid_t group_pid;

            status = setpgid(0, 0);
            if (status)
                exit(EXIT_FAILURE);

            group_pid = getpgrp();
            if (group_pid == -1)
                exit(EXIT_FAILURE);

            ASSERT_EQ(write(write_fd, &group_pid, sizeof(group_pid)), sizeof(group_pid));
            close(read_fd);
            close(write_fd);

            this->test();

            // Quit this process.
            exit(EXIT_SUCCESS);
        }

        void Test::register_test()
        {
            TestMap *test_map = get_test_map();
            ASSERT(test_map->find(t_name) == test_map->end());
            (*test_map)[t_name] = this;
        }
    }
}
